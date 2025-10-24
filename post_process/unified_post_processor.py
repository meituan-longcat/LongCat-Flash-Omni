import datetime
import functools
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio

from longcat_audio_codec.networks.semantic_codec.model_loader import load_decoder


@dataclass
class ProcessedOutput:
    text: Optional[str] = None
    audio_waveform: Optional[torch.Tensor] = None
    audio_codec_tokens: Optional[List[List[int]]] = None


class OmniUnifiedPostProcessor:
    """Unified post-processor for multi-modal outputs (text and audio)"""

    def __init__(
        self,
        config_path: str,
        device: torch.device,
        output_modality: List[str] = ["text", "audio"],
    ):
        """
        Initialize the post-processor

        Args:
            config_path: Path to the decoder configuration file
            device: Torch device (cpu or cuda)
            output_modality: List of output modalities to support ['text', 'audio']
        """
        self.output_modality = output_modality
        self.device = device

        # Initialize processors based on requested modalities
        if "audio" in output_modality:
            self.codec_processor = Codec2WavProcessor(config_path, device)

        # Text processing patterns
        self.special_token_patterns = {
            "bos": r"<longcat_s>",
            "eos": r"</longcat_s>",
            "en_pause": r"\.</pause>",
            "pause": r"</pause>",
        }

    def remove_special_tokens(self, text: str) -> str:
        """
        Remove special tokens from text output

        Args:
            text: Raw text output with special tokens

        Returns:
            Cleaned text without special tokens
        """
        # Replace pause tokens with appropriate punctuation
        text = re.sub(self.special_token_patterns["en_pause"], ". ", text)
        text = re.sub(self.special_token_patterns["pause"], "", text)
        text = re.sub(self.special_token_patterns["eos"], "", text)

        return text.strip()

    def process_text_output(self, text_ids: List[int], tokenizer) -> str:
        """
        Process text token IDs into clean text

        Args:
            text_ids: List of text token IDs
            tokenizer: Tokenizer instance for decoding

        Returns:
            Cleaned text output
        """
        # Filter out invalid tokens
        text_ids = [x for x in text_ids if x > 0]

        # Decode tokens to text
        raw_text = tokenizer.decode(text_ids, skip_special_tokens=False)

        # Remove special tokens
        cleaned_text = self.remove_special_tokens(raw_text)

        return cleaned_text

    def process_audio_output(self, audio_codec_tokens: List[List[int]]) -> np.ndarray:
        """
        Process audio codec tokens into audio waveform

        Args:
            audio_codec_tokens: List of codec token sequences (each sequence has 4 tokens)

        Returns:
            Audio waveform as numpy array (int16)
        """
        if "audio" not in self.output_modality:
            raise ValueError("Audio processing not enabled in output_modality")

        return self.codec_processor.codec2wav(audio_codec_tokens)

    def shift_codec_back(self, original_response: List[List[int]]) -> List[List[int]]:
        unshifted_audio_response = [
            [
                t[0],
                original_response[i + 1][1],
                original_response[i + 1][2],
                original_response[i + 1][3],
            ]
            for i, t in enumerate(original_response[:-1])
            if t[0] != 2
        ]
        return unshifted_audio_response

    def process(self, output: Dict, tokenizer=None) -> ProcessedOutput:
        """
        Main processing method for multi-modal outputs

        Args:
            output: Dictionary containing output data with keys:
                   - output_text_ids: List of text token IDs (optional)
                   - output_multi_ids: List of codec token sequences (optional)
            tokenizer: Tokenizer instance for text decoding (required for text processing)

        Returns:
            ProcessedOutput object containing processed text and/or audio
        """
        result = ProcessedOutput()

        # Process text output if available and requested
        if (
            "text" in self.output_modality
            and "output_ids" in output
            and tokenizer is not None
        ):
            completion_tokens = output['meta_info']['completion_tokens']
            result.text = self.process_text_output(output["output_ids"][-completion_tokens:], tokenizer) # fix for https://github.com/sgl-project/sglang/pull/11384

        # Process audio output if available and requested
        if "audio" in self.output_modality and "aux_info" in output:

            # Shift codec tokens back to original format
            shifted_tokens = self.shift_codec_back(output["aux_info"]["audio_codes"])
            result.audio_codec_tokens = shifted_tokens

            # Convert to audio waveform
            result.audio_waveform = self.process_audio_output(shifted_tokens)

        return result


class Codec2WavProcessor:
    """Processor for converting codec tokens to audio waveforms"""

    def __init__(self, config_path: str, device: torch.device):
        """
        Initialize the codec to waveform processor

        Args:
            config_path: Path to decoder configuration file
            device: Torch device
        """
        self.decoder = load_decoder(config_path, device)
        self.device = device

    def codec2wav(self, codes: List[List[int]]) -> np.ndarray:
        """
        Convert codec tokens to audio waveform

        Args:
            codes: List of codec token (each token should have 4 elements)

        Returns:
            Audio waveform as int16 numpy array
        """

        # Validate input format
        if not all(len(code) == 4 for code in codes):
            raise ValueError("Each codec token must have exactly 4 elements")

        codes_tensor = torch.LongTensor(codes, device="cpu").unsqueeze(0)

        codes_tensor = codes_tensor.transpose(1, 2).to(self.device)

        # Separate semantic tokens (first codebook) and acoustic tokens (remaining 3 codebooks)
        semantic_tokens = codes_tensor[:, 0, :]  # [ 1, sequence_length]
        acoustic_tokens = codes_tensor[:, 1:4, :]  # [ 3, sequence_length]

        # Decode to audio using the decoder
        with torch.no_grad():
            audio = self.decoder(semantic_tokens - 32, acoustic_tokens - 32)

        return audio
