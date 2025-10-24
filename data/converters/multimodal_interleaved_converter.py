#! encoding=utf-8
import json
import math
import re
from typing import Any, Dict

from constants import *
from data.audio_data import AudioFrameLengthCalculator
from global_vars import get_config

from .sft_data_converter import SftDataConverter


def balance_audio_token_during_interleave(decimal):

    decimal = abs(decimal)

    numerator, denominator = decimal.as_integer_ratio()

    num_token_per_chunk = math.floor(numerator / denominator)
    time_sync_interval = denominator if denominator != 1 else None
    return num_token_per_chunk, time_sync_interval


class MultimodalInterleavedConverter(SftDataConverter):
    """
    Processing using streaming audio and video as input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.config = get_config()
        with open(self.config["preprocessor_config"], "r") as f:
            self.processor_config = json.load(f)
        self.codec_pad_id = CODEC_PAD_ID
        self.audio_head_num = self.config["audio_head_num"]

        self.tokenizer = kwargs.get("tokenizer")

        self.frame_length_calculator = AudioFrameLengthCalculator(
            downsample_rate=1, processor_config=self.processor_config
        )

    def convert(
        self, content: Dict[str, Any], text_tokenize_fn, **kwargs
    ) -> Dict[str, Any]:

        c_type = content["type"]
        assert (
            c_type == "multimodal_interleaved"
        ), f"Invalid content type: {c_type}, The required data type is currently multimodal_interleaved."
        assert (
            "video" in content[c_type] and "audio" in content[c_type]
        ), f"Both video and audio fields are required. The currently existing fields are: {list(content.keys())}"

        images = content["multimodal_interleaved"]["video"]
        meta_info = content["meta"]
        if self.config["has_sec_sep"]:
            frame_secs = meta_info.get("frame_secs", [])
        image_token_num = meta_info.get("image_token_num", [])

        frame_sample_strategy = meta_info.get("frame_sample_strategy", "sequential")
        sequential_sample_num_frame_per_sec = meta_info.get(
            "sequential_sample_num_frame_per_sec", 1
        )

        num_video_token = len(images)

        waveform = content["multimodal_interleaved"]["audio"]
        num_audio_token = self.frame_length_calculator(waveform.size(-1))

        # Interleaved audio and video.
        input_text = ""
        if frame_sample_strategy == "sequential":
            audio_duration = num_audio_token * AUDIO_INPUT_FRAME_SEC
            video_duration = num_video_token / sequential_sample_num_frame_per_sec

            num_video_frame_per_chunk = (
                sequential_sample_num_frame_per_sec
                * self.config["omni_streaming_sequential_interval"]
            )
            (
                num_audio_frame_per_chunk,
                time_sync_interval,
            ) = balance_audio_token_during_interleave(
                self.config["omni_streaming_sequential_interval"]
                / AUDIO_INPUT_FRAME_SEC
            )

            num_video_chunk = math.ceil(
                video_duration / self.config["omni_streaming_sequential_interval"]
            )
            num_audio_chunk = math.ceil(
                audio_duration / self.config["omni_streaming_sequential_interval"]
            )

            num_prefill_video_chunk = num_video_chunk - num_audio_chunk
            if num_prefill_video_chunk > 0:
                num_prefill_video_frame = int(
                    num_prefill_video_chunk * num_video_frame_per_chunk
                )

                if self.config["has_sec_sep"]:
                    input_text += "".join(
                        [
                            frame_secs[frame_idx] + f"{DEFAULT_IMAGE_TOKEN}"
                            for frame_idx in range(num_prefill_video_frame)
                        ]
                    )
                else:
                    input_text += DEFAULT_IMAGE_TOKEN * num_prefill_video_frame
            else:
                num_prefill_video_frame = 0

            num_residual_audio_token = num_audio_token
            frame_offset = num_prefill_video_frame
            num_interlaved_chunk = min(num_video_chunk, num_audio_chunk)

            for chunk_idx in range(num_interlaved_chunk):
                require_sync = (
                    (chunk_idx + 1) % time_sync_interval == 0
                    if time_sync_interval is not None
                    else False
                )
                if self.config["has_sec_sep"]:
                    max_frame_idx = len(frame_secs) - 1
                    frame_start_idx = frame_offset + int(
                        chunk_idx * num_video_frame_per_chunk
                    )
                    frame_end_idx = frame_start_idx + num_video_frame_per_chunk
                    input_text += "".join(
                        [
                            frame_secs[min(frame_idx, max_frame_idx)]
                            + f"{DEFAULT_IMAGE_TOKEN}"
                            for frame_idx in range(frame_start_idx, frame_end_idx)
                        ]
                    )
                else:
                    input_text += DEFAULT_IMAGE_TOKEN * num_video_frame_per_chunk

                pad_num_audio_token = (
                    num_audio_frame_per_chunk
                    if not require_sync
                    else num_audio_frame_per_chunk + 1
                )

                if chunk_idx == (num_interlaved_chunk - 1):
                    pad_num_audio_token = num_residual_audio_token

                if chunk_idx == 0:  # first chunk
                    input_text += AUDIO_BOS_TOKEN

                input_text += AUDIO_PAD_TOKEN * pad_num_audio_token

                if chunk_idx == (num_interlaved_chunk - 1):  # last chunk
                    input_text += AUDIO_EOS_TOKEN

                num_residual_audio_token -= pad_num_audio_token

        else:
            num_audio_token_per_video_frame = math.floor(
                num_audio_token / num_video_token
            )
            num_residual_audio_token = num_audio_token
            for chunk_idx in range(num_video_token):
                if self.config["has_sec_sep"]:
                    input_text += frame_secs[chunk_idx] + f"{DEFAULT_IMAGE_TOKEN}"
                else:
                    input_text += DEFAULT_IMAGE_TOKEN

                if chunk_idx == 0:
                    input_text += (
                        AUDIO_BOS_TOKEN
                        + AUDIO_PAD_TOKEN * num_audio_token_per_video_frame
                    )
                elif num_residual_audio_token == num_audio_token_per_video_frame:
                    input_text += (
                        AUDIO_PAD_TOKEN * num_audio_token_per_video_frame
                        + AUDIO_EOS_TOKEN
                    )
                else:
                    input_text += AUDIO_PAD_TOKEN * num_audio_token_per_video_frame

                num_residual_audio_token -= num_audio_token_per_video_frame

            assert (
                num_residual_audio_token >= 0
            ), f"Ensure that the remaining number of audio tokens is greater than 0. The current value is: {num_residual_audio_token}"

            if num_residual_audio_token > 0:
                input_text += (
                    AUDIO_PAD_TOKEN * num_residual_audio_token + AUDIO_EOS_TOKEN
                )

        matches = re.findall(re.escape(DEFAULT_IMAGE_TOKEN), input_text)
        assert len(matches) == len(
            image_token_num
        ), f"Ensure that the number of visual tokens is correct. The current value is: {len(matches)} != {len(image_token_num)}"
        for idx in range(len(image_token_num)):
            num = image_token_num[idx]
            input_text = re.sub(
                re.escape(DEFAULT_IMAGE_TOKEN),
                IMAGE_PAD_TOKEN * num,
                input_text,
                count=1,
            )

        input_ids = text_tokenize_fn(input_text)
        multi_codec_ids = [[self.codec_pad_id] * self.audio_head_num] * len(input_ids)

        return {
            "input_ids": input_ids,
            "input_text": input_text,
            "multi_codec_ids": multi_codec_ids,
            "images": images,
            "audios": waveform,
        }
