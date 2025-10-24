import torch


class AudioFrameLengthCalculator:
    def __init__(self, downsample_rate, processor_config):

        self.downsample_rate = downsample_rate
        self.processor_config = processor_config
        self.window_size = (
            self.processor_config["fbank"]["freq"]
            * self.processor_config["fbank"]["frame_length"]
            // 1000
        )
        self.hop_length = (
            self.processor_config["fbank"]["freq"]
            * self.processor_config["fbank"]["frame_shift"]
            // 1000
        )
        self.slice_stride = self.processor_config["splice"]["stride"]

    def __call__(self, waveform_len):
        frame_len = (waveform_len - self.window_size) // self.hop_length + 1
        frame_len = (frame_len + self.slice_stride - 1) // self.slice_stride
        return frame_len // self.downsample_rate


def freq_masking(feature, n=2, p=0.1):
    batch_size, feat_dim, _ = feature.size()
    max_freq_span = int(p * feat_dim)
    for _ in range(n):
        start_pos = torch.randint(
            0, max(feat_dim - max_freq_span, 1), size=(batch_size, 1)
        )
        span = torch.randint(0, max_freq_span, size=(batch_size, 1))
        end_pos = start_pos + span
        freq_idx = torch.arange(feat_dim).unsqueeze(0).repeat(batch_size, 1)
        freq_mask = ((freq_idx >= start_pos) & (freq_idx < end_pos)).unsqueeze(-1)
        feature.masked_fill_(freq_mask.to(feature.device), 0.0)
    return feature


def time_masking(feature, seq_len, n=2, p=0.05):
    batch_size, _, max_seq_len = feature.size()
    max_time_span = torch.floor(p * seq_len)
    for _ in range(n):
        start_pos = torch.floor(
            (seq_len - max_time_span) * torch.rand(batch_size)
        ).int()
        span = torch.floor(max_time_span * torch.rand(batch_size)).int()
        end_pos = start_pos + span

        time_idx = torch.arange(max_seq_len).unsqueeze(0).repeat(batch_size, 1)
        time_mask = (
            (time_idx >= start_pos.unsqueeze(-1)) & (time_idx < end_pos.unsqueeze(-1))
        ).unsqueeze(1)
        feature.masked_fill_(time_mask.to(feature.device), 0.0)
    return feature
