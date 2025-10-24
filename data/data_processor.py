import json
import math
import os.path as osp
import re

import torch
import yaml

from data.audio_data import AudioFrameLengthCalculator
from data.audio_feature import FeatureProcessor
from data.utils import (
    calculate_token_num_after_post_pooling,
    get_image_token_num,
    load_data,
    unified_collate_fn_for_vlm,
)
from encoders.utils import LongCatImageTransform, load_config
from encoders.vision_adaptor import LongCatVisionConfig
from global_vars import get_config, get_tokenizer

MAX_SAMPLE_NUM_FRAME_PER_VIDEO = 8
AUDIO_INPUT_FRAME_SEC = 0.08


def parse_expression(text):
    pattern = r"<(\w+)_(\d+)>"

    parts = []
    last_end = 0

    for match in re.finditer(pattern, text):
        start, end = match.span()

        if start > last_end:
            text_part = text[last_end:start]
            if text_part:
                parts.append(text_part)

        parts.append(match.group())
        last_end = end

    if last_end < len(text):
        remaining_text = text[last_end:]
        if remaining_text:
            parts.append(remaining_text)

    return parts


def replace_consecutive_pads(text):
    image_pattern = r"(?:<\|image_pad\|>)+"
    audio_pattern = r"(?:<\|audio_pad\|>)+"

    def image_replacement_func(match):
        matched_text = match.group(0)
        count = matched_text.count("<|image_pad|>")
        return f"<image-placeholder={count}>"

    def audio_replacement_func(match):
        matched_text = match.group(0)
        count = matched_text.count("<|audio_pad|>")
        return f"<audio-placeholder={count}>"

    text = re.sub(image_pattern, image_replacement_func, text)
    text = re.sub(audio_pattern, audio_replacement_func, text)
    return text


class DataProcessor:
    def __init__(self, sample_rate=16000):
        config = get_config()
        self.config = config
        with open(config["preprocessor_config"], "r") as f:
            self.processor_config = json.load(f)
            self.processor_config["cmvn"]["global_cmvn"] = osp.join(
                osp.dirname(self.config["preprocessor_config"]), "global_cmvn"
            )

        self.audio_length_calculator = AudioFrameLengthCalculator(
            downsample_rate=1, processor_config=self.processor_config
        )
        self.audio_processor = FeatureProcessor(self.processor_config, training=False)

        self.sample_rate = sample_rate
        univitar_config = LongCatVisionConfig.from_dict(
            load_config(config["vision_config"], v_post_squeeze=True, video_vit=True)
        )
        self.univitar_config = univitar_config
        self.img_transform = LongCatImageTransform(univitar_config)
        self.tokenizer = get_tokenizer()

    def _precess_global_vision_info(self, conversations):
        config = get_config()
        omni_video_fix_post_pooling_times = config["omni_video_fix_post_pooling_times"]
        omni_video_fix_post_pooling_tokens = config[
            "omni_video_fix_post_pooling_tokens"
        ]
        max_vision_length = config["max_vision_length"]

        global_num_imgs, global_imgs, global_image_token_num, global_grid_shapes = (
            [],
            [],
            [],
            [],
        )
        resized_image_token_nums, resized_grid_shapes = [], []

        for tidx, _message in enumerate(conversations):
            print("_message: ", _message)
            role = _message["role"]
            content = _message["content"]
            if role != "user":
                continue

            for i, item in enumerate(content):
                ctype = item["type"]
                if ctype in ["image", "video", "multimodal_interleaved"]:
                    image_list = (
                        item[ctype]["video"]
                        if ctype == "multimodal_interleaved"
                        else item[ctype]
                    )

                    if not isinstance(image_list, list):
                        image_list = [image_list]

                    num_image = len(image_list)
                    print(image_list)
                    global_num_imgs.append(num_image)
                    global_imgs.extend(
                        self.img_transform(
                            image_list, is_video=False if ctype == "image" else True
                        )
                    )
                else:
                    continue

        if len(global_imgs) > 0:
            global_image_token_num, global_grid_shapes = get_image_token_num(
                self.univitar_config, global_imgs, self.tokenizer
            )

        if self.config["v_post_squeeze"] and len(global_imgs) > 0:
            # Calculate the number of visual token.
            max_tokens_video = self.univitar_config.max_tokens_video
            if omni_video_fix_post_pooling_times > 0:
                compress_ratio = omni_video_fix_post_pooling_times
                print(
                    f"> Process video frames uniformly using a fixed compression rate scheme, with a compression rate of {compress_ratio}"
                )
            elif omni_video_fix_post_pooling_tokens > 0:
                fix_compress_ration = round(
                    global_image_token_num[0] / omni_video_fix_post_pooling_tokens, 1
                )
                default_compress_ration = (
                    float(sum(global_image_token_num)) / max_tokens_video
                )
                compress_ratio = max(fix_compress_ration, default_compress_ration)
                print(
                    f"> Process video frames uniformly using a fixed token scheme, with a compression rate of {compress_ratio}, fix_compress_ration={fix_compress_ration}, default_compress_ration={default_compress_ration}"
                )
            else:
                compress_ratio = max(
                    1.0, float(sum(global_image_token_num)) / max_tokens_video
                )
                print(
                    f"> Process uniformly using the default post-compression method, with a maximum number of video tokens of {max_tokens_video}, compress ratio is {compress_ratio}"
                )

            compressed_length = (
                sum(global_image_token_num) / compress_ratio
                if compress_ratio > 1
                else sum(global_image_token_num)
            )
            print(
                f"> The current length of the compressed sequence is: {compressed_length}, The maximum number of visual tokens that the sequence can accept is {max_vision_length}"
            )
            if compressed_length > max_vision_length:
                compress_ratio = (
                    math.ceil(sum(global_image_token_num) / max_vision_length * 10) / 10
                )
                print(
                    f"> The expected compressed length {compressed_length} is still greater than the sequence length {max_vision_length}, so the compression rate is recalculated as: {compress_ratio}"
                )
        else:
            compress_ratio = -1

        vision_idx, vision_offset = 0, 0
        for tidx, _message in enumerate(conversations):
            role = _message["role"]
            content = _message["content"]
            if role != "user":
                continue

            for i, item in enumerate(content):
                ctype = item["type"]
                if ctype in ["image", "video", "multimodal_interleaved"]:
                    num_image = global_num_imgs[vision_idx]
                    start_idx, end_idx = vision_offset, vision_offset + num_image
                    _image_token_nums, _grid_shapes = (
                        global_image_token_num[start_idx:end_idx],
                        global_grid_shapes[start_idx:end_idx],
                    )
                    image_list = global_imgs[start_idx:end_idx]
                    vision_offset += num_image

                    if ctype == "multimodal_interleaved":
                        item[ctype]["video"] = image_list
                    else:
                        item[ctype] = image_list

                    if self.config["v_post_squeeze"]:
                        # compute token numbers and grid shapes.
                        _grid_shapes = [[i[0], i[1], i[2]] for i in _grid_shapes]

                        if omni_video_fix_post_pooling_times > 0:
                            print(
                                f"> The input sequence is expected to be compressed by a fixed factor of {omni_video_fix_post_pooling_times}."
                            )
                        print(
                            f"> Video multi-image grid shape (before post-pooling): {_grid_shapes=}"
                        )
                        print(f"> {_image_token_nums=}")
                        (
                            _grid_shapes,
                            _image_token_nums,
                        ) = calculate_token_num_after_post_pooling(
                            _image_token_nums,
                            _grid_shapes,
                            compress_ratio,
                            self.univitar_config,
                            after_adapter=False,
                        )
                        print(
                            f"> Video multi-image grid shape (after post-pooling): {_grid_shapes=}"
                        )
                        print(f"> {_image_token_nums=}")

                    item["meta"].update(
                        {
                            "image_token_num": _image_token_nums,
                            "grid_shapes": _grid_shapes,
                        }
                    )
                    resized_image_token_nums.append(_image_token_nums)
                    resized_grid_shapes.append(_grid_shapes)

                    conversations[tidx]["content"][i] = item
                    vision_idx += 1
                else:
                    continue
        return (
            conversations,
            compress_ratio,
            resized_image_token_nums,
            resized_grid_shapes,
        )

    def __handle_continuity_audio(self, waveforms):
        """
        Processing of continuous audio
        """
        assert self.audio_processor is not None, "error: audio_processor is None"
        if len(waveforms) > 0:
            waveform_lens = []
            waveform_lens.extend(
                [waveform_tensor.size(-1) for waveform_tensor in waveforms]
            )

            audios = torch.nn.utils.rnn.pad_sequence(
                waveforms, batch_first=True, padding_value=0.0
            )
            all_waveform_lens = torch.LongTensor(waveform_lens).reshape(-1, 1)
            b, t = audios.size()
            audio_mask = torch.arange(t).unsqueeze(0).repeat(b, 1) < all_waveform_lens
            audios, audio_mask = self.audio_processor(audios, audio_mask)
        else:
            audios = None
            audio_mask = None
        return audios, audio_mask

    def __handle_continuity_vision(
        self, images_list, resized_image_token_nums, resized_grid_shapes
    ):
        """
        Processing of vision
        """
        assert (
            len(images_list)
            == len(resized_image_token_nums)
            == len(resized_grid_shapes)
        ), f"error: images_list len{len(images_list)} != resized_image_token_nums len{len(resized_image_token_nums)} != resized_grid_shapes len{len(resized_grid_shapes)}"
        _images_list, _resized_image_token_nums, _resized_grid_shapes = [], [], []
        for imgs, resized_img_token_nums, resized_grid_shape in zip(
            images_list, resized_image_token_nums, resized_grid_shapes
        ):
            if isinstance(imgs, list):
                _images_list.extend(imgs)
                _resized_image_token_nums.extend(resized_img_token_nums)
                _resized_grid_shapes.extend(resized_grid_shape)
            else:
                _images_list.append(imgs)
                _resized_image_token_nums.append(resized_img_token_nums)
                _resized_grid_shapes.append(resized_grid_shape)
        if len(_images_list) > 0:
            img_collate_fn = unified_collate_fn_for_vlm()
            return img_collate_fn(
                [
                    {
                        "image": _images_list,
                        "resized_image_token_nums": _resized_image_token_nums,
                        "resized_grid_shapes": _resized_grid_shapes,
                    }
                ]
            )
        else:
            return {
                "image": None,
                "grid_shapes": None,
                "resized_image_token_nums": None,
                "resized_grid_shapes": None,
            }

    def read_vison_and_audio(self, conversations):
        for cidx, message in enumerate(conversations):
            content = message["content"]

            for i, item in enumerate(content):
                c_type = item["type"]

                if c_type in [
                    "audio",
                    "audio_codec",
                    "image",
                    "video",
                    "multimodal_interleaved",
                ]:
                    try:
                        item = load_data(item)
                    except Exception as e:
                        raise ValueError(
                            f"{item} An exception occurred during data reading. Exception information: {e}"
                        )
                    conversations[cidx]["content"][i] = item
        return conversations

    def process(self, conversations):

        config = get_config()
        conversations = self.read_vison_and_audio(conversations)
        (
            conversations,
            compress_ratio,
            resized_image_token_nums,
            resized_grid_shapes,
        ) = self._precess_global_vision_info(conversations)

        # tokenize
        tokenizer_kwargs = {
            "is_voice_chat": config["is_voice_chat"],
            "output_format": "list",
            "return_pt_tensors": True,
            "infer_insert_assistant": True,
        }
        tokenized_outputs = self.tokenizer.apply_chat_template(
            conversations, **tokenizer_kwargs
        )

        conversations = tokenized_outputs["conversations"]

        input_ids = tokenized_outputs["input_ids"]

        multi_codec_ids = tokenized_outputs["multi_codec_ids"]

        input_ids = torch.cat(
            [input_ids.unsqueeze(-1), multi_codec_ids], dim=-1
        ).unsqueeze(0)
        audios = tokenized_outputs["audios"]
        images = tokenized_outputs["images"]

        audios, audio_mask = self.__handle_continuity_audio(audios)
        images = self.__handle_continuity_vision(
            images, resized_image_token_nums, resized_grid_shapes
        )

        print(
            f"> conversations: {replace_consecutive_pads(json.dumps(conversations, ensure_ascii=False, indent=2))}"
        )

        outputs = {
            "prompts": input_ids,
            "audios": audios,
            "audio_masks": audio_mask,
            "images": images["image"],
            "grid_shapes": images["grid_shapes"],
            "resized_image_token_nums": images["resized_image_token_nums"],
            "resized_grid_shapes": images["resized_grid_shapes"],
        }

        return outputs
