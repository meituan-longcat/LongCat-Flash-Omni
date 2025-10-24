#! encoding=utf-8
import json
from typing import Any, Dict

import torch

from constants import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_PAD_TOKEN,
    CODEC_EOS_ID,
    CODEC_PAD_ID,
    NUM_CODEC_PLACEHOLDERS,
    TMP_PAUSE_TOKEN,
)
from data.audio_data import AudioFrameLengthCalculator
from global_vars import get_config

from .sft_data_converter import SftDataConverter


class AudioSftDataConverter(SftDataConverter):
    """
    Processing using continuous audio as input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = get_config()
        with open(config["preprocessor_config"], "r") as f:
            self.processor_config = json.load(f)

        self.codec_pad_id = CODEC_PAD_ID
        self.audio_head_num = config["audio_head_num"]

        self.frame_length_calculator = AudioFrameLengthCalculator(
            downsample_rate=1, processor_config=self.processor_config
        )

        self.noise_data_list = []
        noise_wav_scp = kwargs.get("noise_wav_scp", None)
        if noise_wav_scp is not None:
            with open(noise_wav_scp, "r") as f:
                for line in f:
                    path = line.strip().split()[-1]
                    self.noise_data_list.append(path)

    def convert(
        self, content: Dict[str, Any], text_tokenize_fn, **kwargs
    ) -> Dict[str, Any]:
        c_type = content["type"]
        assert c_type == "audio"
        main_info = content[c_type]
        meta = content.get("meta", {})
        waveform = main_info

        num_audio_token = self.frame_length_calculator(waveform.size(-1))
        input_text = (
            [AUDIO_BOS_TOKEN] + [AUDIO_PAD_TOKEN] * num_audio_token + [AUDIO_EOS_TOKEN]
        )
        input_text = "".join(input_text)
        multi_codec_ids = [[self.codec_pad_id] * self.audio_head_num] * (
            num_audio_token + 2
        )

        return {
            "input_text": input_text,
            "input_ids": text_tokenize_fn(input_text),
            "multi_codec_ids": multi_codec_ids,
            "audios": waveform,
        }


class AudioCodecSftDataConverter(SftDataConverter):
    """
    Processing with speech codec input
    """

    def __init__(self, **kwargs):
        self.text_pad_token = kwargs.get("pad_token")
        self.text_pad_token_id = kwargs.get("pad_token_id")
        self.conversation_eos_token = kwargs.get("conversation_eos_token")
        assert self.text_pad_token is not None
        assert self.text_pad_token_id is not None
        assert self.conversation_eos_token is not None
        config = get_config()
        self.audio_head_num = config["audio_head_num"]

    def encode_assistant_multi_codec_ids(self, assistant_codec_ids, prefix_len=0):
        if assistant_codec_ids is None:
            return None

        semantic_codec = (
            [CODEC_PAD_ID] * prefix_len
            + [codec[0] + NUM_CODEC_PLACEHOLDERS for codec in assistant_codec_ids]
            + [CODEC_EOS_ID]
        )
        acoustic_list = []
        for i in range(1, self.audio_head_num):
            acoustic_codec = [CODEC_PAD_ID] * (prefix_len + 1) + [
                codec[i] + NUM_CODEC_PLACEHOLDERS for codec in assistant_codec_ids
            ]
            acoustic_list.append(acoustic_codec)
        codec_ids = codec_ids = torch.LongTensor(
            [semantic_codec] + acoustic_list
        ).transpose(0, 1)
        return codec_ids

    def convert(
        self, content: Dict[str, Any], text_tokenize_fn, **kwargs
    ) -> Dict[str, Any]:
        c_type = content["type"]
        assert c_type == "audio_codec"
        main_info = content[c_type]
        meta = content.get("meta", {})

        is_text_pad = kwargs.get("is_text_pad", False)

        codec_ids = meta["codec_ids"]
        content_parts = main_info.split(TMP_PAUSE_TOKEN)

        assert len(content_parts) == len(
            codec_ids
        ), f"Ensure that the number of audio segments is consistent with the number of text segments. {len(content_parts)} != {len(codec_ids)}"

        input_text = ""
        input_ids, multi_codec_ids = [], []
        for part_idx, (content_part, codec_part_ids) in enumerate(
            zip(content_parts, codec_ids)
        ):

            shifted_codec_part_ids = self.encode_assistant_multi_codec_ids(
                codec_part_ids
            )

            multi_codec_ids.extend(shifted_codec_part_ids)

            text_part = content_part
            if part_idx != len(content_parts) - 1:
                text_part += TMP_PAUSE_TOKEN
            else:
                text_part += self.conversation_eos_token

            if not is_text_pad:
                token_id_part = text_tokenize_fn(text_part)
                len_padding_token = shifted_codec_part_ids.size(0) - len(token_id_part)

                assert (
                    len_padding_token >= 0
                ), f"{content_part} has len of {len(token_id_part)}, which > shifed_codec_part_ids len({shifted_codec_part_ids.size(0)})"

                token_id_part += [self.text_pad_token_id] * len_padding_token
                text_part += self.text_pad_token * len_padding_token

                input_text += text_part
                input_ids.extend(token_id_part)
            else:
                input_text += self.text_pad_token * shifted_codec_part_ids.size(0)
                input_ids.extend(
                    [self.text_pad_token_id] * shifted_codec_part_ids.size(0)
                )

        return {
            "input_text": input_text,
            "input_ids": input_ids,
            "multi_codec_ids": multi_codec_ids,
        }

    def degenerate_to_textual_form(self, content):
        for item in content:
            c_type = item["type"]
            if c_type == "audio_codec":
                main_info = item[c_type]
                main_info = main_info.replace(TMP_PAUSE_TOKEN, "")
                item.pop(c_type)
                item["type"] = "text"
                item["text"] = main_info
                item.pop("meta")
        return content
