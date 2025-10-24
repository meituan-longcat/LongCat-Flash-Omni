#! encoding=utf-8
from typing import Any, Dict

from .sft_data_converter import SftDataConverter


class TextSftDataConverter(SftDataConverter):
    """
    Processing using text as input.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert(
        self, content: Dict[str, Any], text_tokenize_fn, **kwargs
    ) -> Dict[str, Any]:
        c_type = content["type"]
        assert c_type == "text"
        main_info = content[c_type]

        input_ids = text_tokenize_fn(main_info)

        return {
            "input_text": main_info,
            "input_ids": input_ids,
        }
