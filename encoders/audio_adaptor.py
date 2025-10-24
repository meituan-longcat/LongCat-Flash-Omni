import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class LongCatAudioConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        self.input_size = kwargs.get("input_size", 1200)
        self.hidden_size = kwargs.get("hidden_size", 6144)
        self.proj_size = kwargs.get("proj_size", 1536)
        self.vocab_size = kwargs.get("vocab_size", 5252)
        self.nlayer = kwargs.get("nlayer", 22)
        self.ndnn = kwargs.get("ndnn", 2)
        self.num_right_layers = kwargs.get("num_right_layers", 6)
        self.left_order = kwargs.get("left_order", 10)
        self.right_order = kwargs.get("right_order", 1)

        self.left_stride = kwargs.get("left_stride", 1)
        self.right_stride = kwargs.get("right_stride", 1)
        self.activation = kwargs.get("activation", "relu6")
        self.dropout = kwargs.get("dropout", 0.1)
        self.layer_drop = kwargs.get("layer_drop", 0.0)
        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        super().__init__(**kwargs)


class LongCatAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        input_size,
        left_kernel_size,
        right_kernel_size,
        dilation=1,
        activation=None,
        dropout=0.0,
    ):
        super(LongCatAudioEncoderLayer, self).__init__()

        self.memory_ln = nn.LayerNorm(input_size)

        self.memory = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=left_kernel_size + right_kernel_size + 1,
            padding=0,
            stride=1,
            dilation=dilation,
            groups=input_size,
            bias=False,
        )

        # Feedforward Net
        self.ffn = nn.Sequential(
            *[
                nn.LayerNorm(input_size),
                nn.Linear(input_size, hidden_size),
                activation,
                nn.Linear(hidden_size, input_size),
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.left_kernel_size = left_kernel_size
        self.right_kernel_size = right_kernel_size
        self.dilation = dilation
        self.input_size = input_size
        self.hidden_size = hidden_size

    @property
    def ops_count(self):
        return self.ops

    def forward(self, input_feat):
        # input (B, T, H)
        residual = input_feat

        # dfsmn-memory
        memory_input = (
            self.memory_ln(input_feat).transpose(1, 2).contiguous()
        )  # (B, H, T)
        pad_input_fea = F.pad(
            memory_input,
            (
                self.left_kernel_size * self.dilation,
                self.right_kernel_size * self.dilation,
                0,
                0,
            ),
        )  # (B,N,T+(l+r)*d)

        memory_out = self.memory(pad_input_fea).transpose(1, 2).contiguous()
        memory_out = self.dropout(memory_out) + residual

        M = memory_out.size(0) * memory_out.size(1)

        residual = memory_out
        fc_output = self.ffn(memory_out)
        output = fc_output + residual

        return output


class LongCatAudioEncoder(torch.nn.Module):
    """Feedforward Sequential Memory Network(FSMN)

    DFSMN acoustic model, followed by arbitrary number of DNN layer.

    Parameters:
        input_size (int): input dimension size
        hidden_size (int): FSMN hidden size
        proj_size (int): FSMN projection size
        vocab_size (int): output dimension for softmax
        nlayer (int): number of DFSMN layers
        ndnn (int): number of the dnn layers following dfsmn (including
                    output layer)
        lo (int): left order
        ro (int): right order
        ls (int): left stride
        rs (int): right stride
        activation (str): ``relu`` ``relu2`` or ``relu6``. The default activation
                          in FSMN and DNN, empirically ``relu6`` trains more stable
                          than ``relu``, especially in small dataset.
                          ``relu2`` is recommended for local model.
    """

    def __init__(self, config: LongCatAudioConfig):
        super().__init__()
        self.config = config
        self.layer_drop = config.layer_drop
        self.gradient_checkpointing = config.gradient_checkpointing
        assert self.layer_drop == 0.0
        if config.activation == "relu6":
            self.activation = nn.ReLU6(inplace=True)
        elif config.activation == "relu2":
            self.activation = nn.Hardtanh(min_val=0, max_val=2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.ffn = nn.Sequential(
            *[
                nn.Linear(config.input_size, config.hidden_size),
                self.activation,
                nn.Linear(config.hidden_size, config.proj_size, bias=False),
            ]
        )

        self.fsmn_layers = nn.ModuleList(
            [
                LongCatAudioEncoderLayer(
                    config.hidden_size,
                    config.proj_size,
                    config.left_order,
                    (
                        0
                        if i < (config.nlayer - config.num_right_layers)
                        else config.right_order
                    ),
                    config.left_stride,
                    activation=self.activation,
                    dropout=config.dropout,
                )
                for i in range(config.nlayer)
            ]
        )
        if config.ndnn > 0:
            dnns = [nn.Linear(config.proj_size, config.hidden_size), self.activation]
            for _ in range(config.ndnn - 1):
                dnns.extend(
                    [nn.Linear(config.hidden_size, config.hidden_size), self.activation]
                )
            output_in = config.hidden_size
        elif config.ndnn == 0:
            dnns = []
            output_in = config.proj_size
        else:
            raise ValueError("This implement needs ndnn >= 0 followed by FSMN layers")
        self.dnn = nn.Sequential(*dnns)
        self.output = nn.Linear(output_in, config.vocab_size)

    @property
    def output_embedding_dim(self):
        return (
            self.config.proj_size if self.config.ndnn == 0 else self.config.hidden_size
        )

    def forward(self, input_features, attention_mask=None, return_logits=False):
        # (B, T, C)
        attention_mask_int = attention_mask.unsqueeze(-1).int()
        hidden_states = self.ffn(input_features)
        for layer in self.fsmn_layers:
            hidden_states = hidden_states * attention_mask_int
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states * attention_mask_int
        hidden_states = self.dnn(hidden_states)
        hidden_states = hidden_states + 0.0 * self.output(hidden_states).sum()

        if return_logits:
            return self.output(hidden_states), attention_mask
        else:
            return hidden_states, attention_mask


class LongCatOmniAudioAdaptor(torch.nn.Module):
    def __init__(self, config_path: LongCatAudioConfig, proj_output_dim: int):
        super().__init__()
        config_dict = json.load(open(config_path, "r", encoding="utf8"))
        config = LongCatAudioConfig.from_dict(config_dict)

        self.emb_mult = 1.0

        self.audio_encoder = LongCatAudioEncoder(config)

        self.audio_projector = nn.Sequential(
            nn.Linear(self.audio_encoder.output_embedding_dim, proj_output_dim),
            nn.GELU(),
            nn.Linear(proj_output_dim, proj_output_dim),
        )

    def forward(self, audios_features, audio_feature_masks):
        audio_embs, audio_mask = self.audio_encoder(
            audios_features.to(torch.bfloat16), audio_feature_masks
        )
        audio_embs = self.audio_projector(audio_embs)
        audio_embs = audio_embs * self.emb_mult

        # Varlen Should be cat
        if audio_embs.size(0) > 1:
            audio_embs = torch.cat([i for i in audio_embs], dim=0).unsqueeze(0)
            audio_mask = torch.cat([i for i in audio_mask], dim=0).unsqueeze(0)

        return audio_embs[audio_mask]
