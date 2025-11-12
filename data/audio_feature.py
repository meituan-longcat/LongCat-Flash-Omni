import math
import random
import sys

import kaldiio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.audio_data import freq_masking, time_masking

MEL_SCALE_FACTOR = 1127.0
FREQ_SCALE_FACTOR = 700.0
DFSMN_FEAT_DIM = 1200
WINDOW_SCALE_FACTOR = 1000


def calculate_mean_var_postprocess(cmvn_mean, cmvn_var):
    cmvn_mean = torch.from_numpy(cmvn_mean).float()
    cmvn_var = torch.from_numpy(cmvn_var).float()
    eps = torch.tensor(torch.finfo(torch.float32).eps)
    cmvn_scale = 1.0 / torch.max(eps, torch.sqrt(cmvn_var))
    return cmvn_mean, cmvn_scale


def generate_padding_mask_by_size(max_length, lengths, device="cpu"):
    mask = None
    if lengths is not None:
        lengths = torch.as_tensor(lengths)
        mask = (
            lengths[:, None] <= torch.arange(max_length, device=lengths.device)[None, :]
        )
        mask = mask.to(device)
    return mask


def calculate_mean_var(cmvn_stats_0, cmvn_stats_1, cmvn_count):
    cmvn_mean = cmvn_stats_0 / cmvn_count
    cmvn_var = cmvn_stats_1 / cmvn_count - cmvn_mean * cmvn_mean
    return cmvn_mean, cmvn_var


def next_power_of_2(x: int) -> int:
    # return 1 if x == 0 else 2 ** (x - 1).bit_length()
    return 1 if x == 0 else 1 << (x - 1).bit_length()


def mel_scale_scalar(freq: float) -> float:
    return MEL_SCALE_FACTOR * math.log(1.0 + freq / FREQ_SCALE_FACTOR)


def mel_scale(freq: Tensor) -> Tensor:
    return MEL_SCALE_FACTOR * (1.0 + freq / FREQ_SCALE_FACTOR).log()


def convert_to_same_padding(feats: Tensor, input_lens: Tensor) -> Tensor:
    # convert feats to "same" padding
    B, T, D = feats.size()
    last_idx = input_lens - 1 + torch.arange(B, device=feats.device) * T
    # last_idx = input_lens.to(feats.device) + torch.arange(-1, B * T - 1, T, device=feats.device)
    backgroud = feats.reshape(B * T, D)[last_idx].unsqueeze(1).repeat(1, T, 1)
    mask = generate_padding_mask_by_size(T, input_lens, feats.device).unsqueeze(2)
    feats = feats * (~mask) + backgroud * mask
    return feats


class Fbank(nn.Module):
    """
    class for GPU batch fbank computation
    modified from https://github.com/pytorch/audio/blob/main/torchaudio/compliance/kaldi.py#L514
    main modifications:
        1. support batch computation
        2. pre-compute filter banks
        3. simplify the process and remove process we do not use
    refer to kaldi for the meaning of the config parameters
    """

    def __init__(self, conf):
        super(Fbank, self).__init__()
        self.dither = conf["dither"]
        self.frame_length = frame_length = conf["frame_length"]
        self.frame_shift = frame_shift = conf["frame_shift"]
        self.preemphasis = preemphasis = conf["preemphasis"]
        self.freq = freq = conf["freq"]
        high_freq = conf["high_freq"]
        low_freq = conf["low_freq"]
        self.num_mel_bins = num_mel_bins = conf["num_mel_bins"]

        assert self.freq in [8000, 16000], "freq should be 8000 or 16000"

        self.window_shift = window_shift = int(
            self.freq * self.frame_shift / WINDOW_SCALE_FACTOR
        )
        self.window_size = window_size = int(
            self.freq * self.frame_length / WINDOW_SCALE_FACTOR
        )
        self.padded_window_size = padded_window_size = next_power_of_2(self.window_size)

        window = torch.hann_window(
            self.window_size, periodic=False, dtype=torch.float32
        ).pow(0.85)
        self.register_buffer("window", window)

        # Get mel filter banks
        num_fft_bins = padded_window_size // 2
        nyquist = freq / 2
        self.high_freq = high_freq = high_freq + nyquist if high_freq < 0 else high_freq
        fft_bin_width = freq / padded_window_size

        mel_low_freq = mel_scale_scalar(low_freq)
        mel_high_freq = mel_scale_scalar(high_freq)
        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_mel_bins + 1)
        bins = torch.arange(num_mel_bins).unsqueeze(1)
        left_mel = mel_low_freq + bins * mel_freq_delta
        center_mel = mel_low_freq + (bins + 1.0) * mel_freq_delta
        right_mel = mel_low_freq + (bins + 2.0) * mel_freq_delta

        mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

        # size (num_mel_bins, num_fft_bins)
        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)
        mel_banks = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
        mel_banks = torch.nn.functional.pad(mel_banks, (0, 1), mode="constant", value=0)
        mel_banks = mel_banks.t()
        self.register_buffer("mel_banks", mel_banks)

        eps = torch.tensor(torch.finfo(torch.float32).eps)
        self.register_buffer("eps", eps)

        # need padding for lower feat
        self.pad_size = 0
        if "padded_num_mel_bins" in conf:
            self.pad_size = conf["padded_num_mel_bins"] - self.num_mel_bins

    def forward(self, batch_wav, input_lens):
        """
        Args:
            batch_wav: batched wav, shape (batch, wav_len)
            input_lens: shape (batch,)
        Returns:
            batch_fbank: shape (batch, frame_len, num_mel_bins)
            output_lens: shape (batch, )
        """
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                b, wav_len = batch_wav.size()

                if wav_len < self.window_size:
                    feats = torch.empty(
                        (b, 0, self.num_mel_bins),
                        dtype=batch_wav.dtype,
                        device=batch_wav.device,
                    )
                    input_lens = torch.zeros(
                        b, dtype=input_lens.dtype, device=input_lens.device
                    )
                    return feats, input_lens

                frame_len = 1 + (wav_len - self.window_size) // self.window_shift
                # batch_wav should be contiguous before using as_strided
                batch_wav = batch_wav.contiguous()
                # (n, frame_len, window_size)
                frames = batch_wav.as_strided(
                    (b, frame_len, self.window_size), (wav_len, self.window_shift, 1)
                )
                if self.dither != 0.0:
                    rand_gauss = torch.randn(
                        frames.shape, device=frames.device, dtype=frames.dtype
                    )
                    frames = frames + rand_gauss * self.dither

                # (n, frame_len, window_size + 1)
                padded = torch.nn.functional.pad(frames, (1, 0), mode="replicate")
                frames = frames - self.preemphasis * padded[:, :, :-1]
                frames = frames * self.window
                if self.padded_window_size != self.window_size:
                    padding_right = self.padded_window_size - self.window_size
                    frames = torch.nn.functional.pad(
                        frames, (0, padding_right), mode="constant", value=0
                    )
                # spectrum
                spec = torch.fft.rfft(frames).abs()
                # power, (n, frame_len, num_fft_bins)
                spec = spec.pow(2)

                mel_energy = torch.matmul(spec, self.mel_banks)
                mel_energy = torch.max(mel_energy, self.eps).log()
                # input_lens = (input_lens - self.window_size) // self.window_shift + 1
                input_lens = (
                    torch.div(
                        input_lens - self.window_size,
                        self.window_shift,
                        rounding_mode="trunc",
                    )
                    + 1
                )

                # padding feature when mixband is 8k
                if self.pad_size > 0:
                    mel_energy = F.pad(mel_energy, (0, self.pad_size))

                return mel_energy, input_lens


class Cmvn(nn.Module):
    """
    class for GPU cmvn computation
    """

    def __init__(self, conf):
        super(Cmvn, self).__init__()

        self.skip_computation = False
        if conf["global_cmvn"] == "":
            self.skip_computation = True
            return

        cmvn_stats = kaldiio.load_mat(conf["global_cmvn"])
        cmvn_count = cmvn_stats[0][-1]

        cmvn_mean, cmvn_var = calculate_mean_var(
            cmvn_stats[0][:-1], cmvn_stats[1][:-1], cmvn_count
        )
        cmvn_mean, cmvn_scale = calculate_mean_var_postprocess(cmvn_mean, cmvn_var)

        self.register_buffer("cmvn_mean", cmvn_mean)
        self.register_buffer("cmvn_scale", cmvn_scale)

    def forward(self, feats, input_lens):
        if self.skip_computation:
            return feats, input_lens

        B, T, D = feats.size()
        if T == 0:
            return feats, input_lens

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                feats = (feats - self.cmvn_mean) * self.cmvn_scale
                return feats, input_lens


class Splice(nn.Module):
    """
    class for GPU splice computation
    """

    def __init__(self, conf):
        super(Splice, self).__init__()
        self.left = conf["left"]
        self.right = conf["right"]
        self.stride = conf["stride"]
        self.random_start = conf["random_start"]
        self.seed = conf["seed"]
        self.random = random.Random()
        self.random.seed(self.seed)
        self.skip_computation = False

        if self.left == 0 and self.right == 0 and self.stride == 1:
            self.skip_computation = True

    def forward(self, feats, input_lens):
        if self.skip_computation:
            return feats, input_lens

        B, T, D = feats.size()
        if T == 0:
            feats = torch.empty(
                (B, T, D * (self.left + self.right + 1)),
                dtype=feats.dtype,
                device=feats.device,
            )
            return feats, input_lens

        if self.random_start:
            left = self.left - self.random.randint(0, self.stride - 1)
            right = self.right + self.left - left
        else:
            left = self.left
            right = self.right

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                # convert feats to same padding
                feats = convert_to_same_padding(feats, input_lens)
                m = left + right + 1
                left_pad = feats[:, 0:1].repeat(1, left, 1)
                right_pad = feats[:, T - 1 :].repeat(1, right, 1)
                feats = torch.cat((left_pad, feats, right_pad), dim=1)
                feats = feats.contiguous()

                T_pad = T + left + right
                T_out = (T + self.stride - 1) // self.stride
                feats = torch.as_strided(
                    feats, (B, T_out, D * m), (T_pad * D, self.stride * D, 1)
                )
                feats = feats.contiguous()
                # input_lens = (input_lens + self.stride - 1) // self.stride
                input_lens = torch.div(
                    input_lens + self.stride - 1, self.stride, rounding_mode="trunc"
                )
                return feats, input_lens


class Delta(nn.Module):
    """
    class for delta computation
    """

    def __init__(self, conf):
        super(Delta, self).__init__()
        self.delta_order = conf["delta_order"]
        self.window_size = conf["window_size"]
        self.skip_computation = False
        if self.delta_order == 0:
            self.skip_computation = True
            return

        n = self.window_size
        denom = n * (n + 1) * (2 * n + 1) / 3
        base_kernel = np.arange(-n, n + 1) / denom
        scales = [np.asarray([1.0])]
        for order in range(1, self.delta_order + 1):
            prev_scale = scales[-1]
            cur_scale = np.convolve(prev_scale, base_kernel)
            scales.append(cur_scale)

        scales = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(x) for x in scales], batch_first=True
        ).float()
        self.register_buffer("scales", scales)

    def forward(self, feats, input_lens):
        """
        Args:
            feats: shape (batch, time0, dim0)
            input_lens: shape (batch,)
        Returns:
            feats: shape (batch, time0, dim0 * (delta_order + 1))
            output_lens: shape (batch,)
        """
        if self.skip_computation:
            return feats, input_lens

        B, T, D = feats.size()
        if T == 0:
            feats = torch.empty(
                (B, T, D * (self.delta_order + 1)),
                dtype=feats.dtype,
                device=feats.device,
            )
            return feats, input_lens

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                feats = convert_to_same_padding(feats, input_lens)
                # (B, T, D) -> (B, D, T)
                feats = feats.transpose(1, 2)
                partial_res = [feats]

                feats = feats.reshape(1, -1, T)
                for i in range(1, self.delta_order + 1):
                    n = i * self.window_size * 2 + 1
                    scale = self.scales[i, : 2 * i * self.window_size + 1]
                    scale = scale.repeat(feats.shape[1], 1, 1)
                    feats = F.pad(
                        feats, (self.window_size, self.window_size), mode="replicate"
                    )
                    delta_feats = F.conv1d(feats, scale, groups=feats.shape[1])
                    partial_res.append(delta_feats.reshape(B, D, T))

                feats = torch.concat(partial_res, dim=1)
                feats = feats.transpose(1, 2)

                return feats, input_lens


class Splice(nn.Module):
    """
    class for GPU splice computation
    """

    def __init__(self, conf):
        super(Splice, self).__init__()
        self.left = conf["left"]
        self.right = conf["right"]
        self.stride = conf["stride"]
        self.random_start = conf["random_start"]
        self.seed = conf["seed"]
        self.random = random.Random()
        self.random.seed(self.seed)
        self.skip_computation = False

        if self.left == 0 and self.right == 0 and self.stride == 1:
            self.skip_computation = True

    def forward(self, feats, input_lens):
        if self.skip_computation:
            return feats, input_lens

        B, T, D = feats.size()
        if T == 0:
            feats = torch.empty(
                (B, T, D * (self.left + self.right + 1)),
                dtype=feats.dtype,
                device=feats.device,
            )
            return feats, input_lens

        if self.random_start:
            left = self.left - self.random.randint(0, self.stride - 1)
            right = self.right + self.left - left
        else:
            left = self.left
            right = self.right

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                # convert feats to same padding
                feats = convert_to_same_padding(feats, input_lens)
                m = left + right + 1
                left_pad = feats[:, 0:1].repeat(1, left, 1)
                right_pad = feats[:, T - 1 :].repeat(1, right, 1)
                feats = torch.cat((left_pad, feats, right_pad), dim=1)
                feats = feats.contiguous()

                T_pad = T + left + right
                T_out = (T + self.stride - 1) // self.stride
                feats = torch.as_strided(
                    feats, (B, T_out, D * m), (T_pad * D, self.stride * D, 1)
                )
                feats = feats.contiguous()
                input_lens = torch.div(
                    input_lens + self.stride - 1, self.stride, rounding_mode="trunc"
                )
                return feats, input_lens


class FeatureProcessor:
    """
    class for mixband feature computation which include fbank and cmvn
    """

    def __init__(self, processor_config, training=False, use_spec_augment=False):

        self.fbank = Fbank(processor_config["fbank"])
        self.delta = Delta(processor_config["delta"])
        self.cmvn = Cmvn(processor_config["cmvn"])
        self.splice = (
            Splice(processor_config["splice"]) if "splice" in processor_config else None
        )

        self.training = training
        self.use_spec_augment = use_spec_augment

        self.feature_dim = DFSMN_FEAT_DIM

    def __call__(self, waveforms, waveform_mask=None):
        input_lens = torch.sum(waveform_mask, dim=-1)
        with torch.no_grad():
            feat, input_lens = self.fbank(waveforms * 32768, input_lens)
            feat, input_lens = self.delta(feat, input_lens)
            feat, input_lens = self.cmvn(feat, input_lens)

            if self.training and self.use_spec_augment:
                feat = freq_masking(feat, n=2, p=0.1)
                feat = time_masking(feat, input_lens, n=2, p=0.1)

            if self.splice is not None:
                feat, input_lens = self.splice(feat, input_lens)

        B, T, _ = feat.size()
        feat_mask = torch.arange(T).expand(B, T) < input_lens.unsqueeze(-1)
        return feat, feat_mask
