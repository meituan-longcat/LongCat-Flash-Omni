#! encoding=utf-8
from typing import Any, Dict

from constants import CODEC_PAD_ID, IMAGE_PAD_TOKEN
from global_vars import get_config

from .sft_data_converter import SftDataConverter


class ImageSftDataConverter(SftDataConverter):
    """
    Processing using vision as input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = get_config()
        self.tokenizer = kwargs.get("tokenizer")

    def convert(
        self, content: Dict[str, Any], text_tokenize_fn, **kwargs
    ) -> Dict[str, Any]:
        c_type = content["type"]
        assert c_type == "image" or c_type == "video"
        processed_images = content[c_type]
        meta = content.get("meta", {})
        _image_token_num_list = meta.get("image_token_num", [])

        if c_type == "image":
            assert len(processed_images) == 1, f"Only single image is supported."
            input_text = "".join([IMAGE_PAD_TOKEN] * _image_token_num_list[0]) + "\n"
        else:
            frame_secs = meta.get("frame_secs", ["" * len(processed_images)])
            assert len(frame_secs) == len(
                processed_images
            ), f"The number of video frames does not match the video duration, {len(frame_secs)} != {len(processed_images)}"
            assert len(processed_images) == len(_image_token_num_list)
            input_text = ""
            for _sec, _image_token_num in zip(frame_secs, _image_token_num_list):
                if self.config["has_sec_sep"]:
                    input_text += _sec + "".join([IMAGE_PAD_TOKEN] * _image_token_num)
                else:
                    input_text += "".join([IMAGE_PAD_TOKEN] * _image_token_num)

        input_ids = text_tokenize_fn(input_text)

        return {
            "input_ids": input_ids,
            "input_text": input_text,
            "images": processed_images,
        }
