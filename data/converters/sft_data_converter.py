#! encoding=utf-8

from abc import ABC, abstractmethod
from typing import Any, Dict


class SftDataConverter(ABC):
    def __init__(self, **kwargs):
        pass

    """
    Interface class for multimodal data processing.
    """

    @abstractmethod
    def convert(
        self, content: Dict[str, Any], text_tokenize_fn, **kwargs
    ) -> Dict[str, Any]:
        raise Exception("convert() method not implement yet.")
