import itertools
import json
import os
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from PIL import Image
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

    def forward(self, grid_shapes, spatial_merge_size=2):
        pos_ids = []
        s = spatial_merge_size
        for t, h, w in grid_shapes:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(h // s, s, w // s, s)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(h // s, s, w // s, s)
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = torch.tensor(grid_shapes).max()
        rotary_pos_emb_full = self.forward_(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb


class FlashAttention(nn.Module):
    # https://github.com/Dao-AILab/flash-attention/blob/v0.2.8/flash_attn/flash_attention.py
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(
        self,
        q,
        k,
        v,
        causal=False,
        cu_seqlens=None,
        max_seqlen=None,
        need_weights=False,
    ):
        assert not need_weights
        assert max_seqlen is not None
        assert cu_seqlens is not None

        output, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            0.0,
            causal,
            return_debug_mask=False,
        )
        return output, None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LongCatVisionConfig(PretrainedConfig):
    def __init__(
        self,
        resolution_mode="native",
        init_method="xavier",
        num_channels=3,
        patch_size=14,
        temporal_patch_size=2,
        image_size=1792,
        patch_dropout=0.0,
        attention_dropout=0.0,
        dropout=0.0,
        drop_path_rate=0.0,
        initializer_range=1e-10,
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_size=1024,
        intermediate_size=4224,
        patch_embedding_bias=True,
        qk_normalization=True,
        qkv_bias=False,
        initializer_factor=0.1,
        use_pre_norm=False,
        pe_type="rope2d",
        rope_theta=10000,
        spatial_merge_size=1,
        norm_type="RMSNorm",
        hidden_act="SwiGLU",
        use_flash_attn=True,
        layer_norm_eps=1e-6,
        min_tokens=576,
        max_tokens=16384,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        v_post_squeeze=False,
        adaptor_dim=7168,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resolution_mode = resolution_mode
        self.init_method = init_method
        self.pe_type = pe_type
        self.rope_theta = rope_theta
        self.temporal_patch_size = temporal_patch_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_dropout = patch_dropout
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.patch_embedding_bias = patch_embedding_bias
        self.qk_normalization = qk_normalization
        self.qkv_bias = qkv_bias
        self.initializer_factor = initializer_factor
        self.use_pre_norm = use_pre_norm
        self.norm_type = norm_type
        self.hidden_act = hidden_act
        self.use_flash_attn = use_flash_attn
        self.layer_norm_eps = layer_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.image_mean = image_mean
        self.image_std = image_std
        self.v_post_squeeze = v_post_squeeze
        self.adaptor_dim = adaptor_dim

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        if "vision_config" in config_dict:
            config_dict = config_dict["vision_config"]

        if (
            "model_type" in config_dict
            and hasattr(cls, "model_type")
            and config_dict["model_type"] != cls.model_type
        ):
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class SwiGLU(nn.Module):
    def __init__(self, config: LongCatVisionConfig):
        super().__init__()
        self.config = config
        self.inner_hidden_size = int(config.intermediate_size * 2 / 3)
        self.act = ACT2FN["silu"]
        self.fc1 = nn.Linear(config.hidden_size, self.inner_hidden_size)
        self.fc2 = nn.Linear(self.inner_hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, self.inner_hidden_size)
        self.norm = RMSNorm(self.inner_hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(x)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(self.norm(hidden_states * self.fc3(x)))
        return hidden_states


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LongCatVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert config.use_flash_attn is True, "FlashAttention must be used!"
        assert self.head_dim * self.num_heads == self.embed_dim

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        if self.config.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        assert hidden_states.ndim == 2
        cu_seqlens = kwargs["cu_seqlens"]
        max_seqlen = kwargs["max_seqlen"]

        qkv = self.qkv(hidden_states)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, h=self.num_heads
        )
        bind_dim = qkv.dim() - 3
        q, k, v = qkv.unbind(bind_dim)
        rotary_pos_emb = kwargs["rotary_pos_emb"]
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        if self.config.qk_normalization:
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
        context, _ = self.inner_attn(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, causal=False
        )
        # shape: [s, h, d]
        context = rearrange(context, "... h d -> ... (h d)")
        hidden_states = self.proj(context)
        return hidden_states


class LongCatVisionEmbeddings(nn.Module):
    def __init__(self, config: LongCatVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.use_bias = config.patch_embedding_bias
        self.patch_embedding = nn.Conv3d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            bias=self.use_bias,
        )

    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> torch.Tensor:
        pixel_values = pixel_values.view(-1, 3, *self.kernel_size)
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


class LongCatVisionEncoderLayer(nn.Module):
    def __init__(self, config: LongCatVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        assert config.hidden_act == "SwiGLU"

        self.attn = Attention(config)
        self.norm1 = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SwiGLU(config)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))

    def forward(self, hidden_states: torch.Tensor, extra_layer_args):
        norm = self.norm1(hidden_states)
        atten = self.attn(norm, **extra_layer_args)
        hidden_states = hidden_states + atten * self.ls1
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2
        return hidden_states


class LongCatVisionEncoder(nn.Module):
    """Transformer encoder consisting of `config.num_hidden_layers` self attention layers."""

    def __init__(self, config: LongCatVisionConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [
                LongCatVisionEncoderLayer(config)
                for idx in range(config.num_hidden_layers)
            ]
        )

        assert self.config.pe_type == "rope2d"
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding2D(
            head_dim // 2, theta=self.config.rope_theta
        )

    def forward(self, hidden_states: torch.Tensor, extra_layer_args: dict):
        extra_layer_args = dict(extra_layer_args)
        rotary_pos_emb: torch.Tensor = self.rotary_pos_emb(
            extra_layer_args["grid_shapes"], self.config.spatial_merge_size
        )
        extra_layer_args["rotary_pos_emb"] = rotary_pos_emb
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states, extra_layer_args=extra_layer_args
            )
        return hidden_states


class LongCatVisionModel(torch.nn.Module):
    def __init__(self, config: LongCatVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = LongCatVisionEmbeddings(config)
        self.encoder = LongCatVisionEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_tensor(self, input_tensor):
        """Set model chunk input tensor."""

        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for llava"
        self.encoder_hidden_state = input_tensor[0]

    def get_cu_seqlens(self, grid_shapes):
        token_num_per_image = [t * h * w for t, h, w in grid_shapes]
        max_seqlen = max(token_num_per_image)
        cu_seqlens = list(itertools.accumulate([0] + token_num_per_image))
        cu_seqlens_np = np.array(cu_seqlens)
        cu_seqlens = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=torch.cuda.current_device()
        )
        return cu_seqlens, cu_seqlens_np, max_seqlen

    def forward(self, pixel_values: torch.Tensor, grid_shapes: List[int]):
        assert (
            len(pixel_values.shape) == 2
        ), "The input must be a two-dimensional Tensor (ndim=2) of shape (num_tokens, dim)"
        assert grid_shapes is not None
        assert all(w % 2 == 0 for _, _, w in grid_shapes)

        cu_seqlens, cu_seqlens_np, max_seqlen = self.get_cu_seqlens(grid_shapes)
        hidden_states = self.embeddings(pixel_values)
        extra_layer_args = dict(
            grid_shapes=grid_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = self.encoder(hidden_states, extra_layer_args)
        return hidden_states


class VisionProjector(nn.Module):
    def __init__(self, encoder_dim, adapter_dim, proj_output_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.p0 = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(adapter_dim, proj_output_dim)

    def forward(self, hidden_states: torch.Tensor, grid_shapes):
        hidden_states_list = []
        start_idx = 0
        for grid_shape in grid_shapes:
            t, h, w = grid_shape
            assert t == 1
            assert not w % 2, "w need divided by 2"
            end_idx = start_idx + h * w
            hidden_state = hidden_states[start_idx:end_idx]
            start_idx = end_idx
            hidden_state = hidden_state.reshape(h, w, -1)
            hidden_state = hidden_state.reshape(h, int(w * 0.5), self.encoder_dim)
            hidden_state = hidden_state.reshape(-1, self.encoder_dim)
            hidden_state = self.proj(self.p0(hidden_state))
            hidden_states_list.append(hidden_state)
        hidden_states_ret = torch.concatenate(hidden_states_list, dim=0)
        return hidden_states_ret


class LongCatOmniVisionAdaptor(torch.nn.Module):
    def __init__(self, config_path: str, proj_output_dim: int):
        super().__init__()
        config_dict = json.load(open(config_path, "r", encoding="utf8"))
        config = LongCatVisionConfig.from_dict(config_dict)
        self.config = config

        self.vision_encoder = LongCatVisionModel(config)
        self.vision_projector = VisionProjector(
            config.hidden_size * 2, config.adaptor_dim, proj_output_dim
        )

    def forward(self, pixel_values, grid_shapes: list):
        hidden_states = self.vision_encoder(pixel_values, grid_shapes)
        hidden_states = self.vision_projector(hidden_states, grid_shapes)
        return hidden_states
