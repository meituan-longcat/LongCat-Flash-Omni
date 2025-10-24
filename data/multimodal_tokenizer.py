import logging
import re
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from constants import *
from data.converters import *
from global_vars import get_config

logger = logging.getLogger(__file__)


def replace_consecutive_pads(text):
    image_pattern = r"(?:<\|image_pad\|>)+"
    audio_pattern = r"(?:<\|audio_pad\|>)+"
    text_pattern = r"(?:<pad>)+"

    def image_replacement_func(match):
        matched_text = match.group(0)
        count = matched_text.count("<|image_pad|>")
        return f"<image-placeholder={count}>"

    def audio_replacement_func(match):
        matched_text = match.group(0)
        count = matched_text.count("<|audio_pad|>")
        return f"<audio-placeholder={count}>"

    def text_replacement_func(match):
        matched_text = match.group(0)
        count = matched_text.count("<pad>")
        return f"<textpad-placeholder={count}>"

    text = re.sub(image_pattern, image_replacement_func, text)
    text = re.sub(audio_pattern, audio_replacement_func, text)
    text = re.sub(text_pattern, text_replacement_func, text)
    return text


class MultimodalTokenizer:
    def __init__(self, tokenizer_name_or_path, **kwargs):
        """Initialize the multimodal tokenizer
        Args:
            tokenizer_name_or_path: Loading path of the original text modality tokenizer,
            used to add additional special tokens for multimodal model.
        """

        self.config = get_config()
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self.padding_side = "right"
        kwargs.update({"padding_side": self.padding_side})
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True, **kwargs
        )

        print(
            f"> Successfully loaded the tokenizer from the path {tokenizer_name_or_path}."
        )
        print(f"> Vocabulary size = {len(self.get_vocab())}")
        self.tokenizer.padding_side = self.padding_side
        print(f"> padding_side = {self.padding_side}")

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eod = self.tokenizer.eos_token_id

        self.bos_token = (
            self.tokenizer.bos_token if self.tokenizer.bos_token else self.eos_token
        )
        self.bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token
            else self.eos_token_id
        )

        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        kwargs.update({"pad_token": self.pad_token})
        kwargs.update({"pad_token_id": self.pad_token_id})

        # List of extended special tokens, including audio, system, user, assistant, and other related marker
        self.extended_special_tokens_list = [
            AUDIO_BOS_TOKEN,
            AUDIO_EOS_TOKEN,
            AUDIO_PAD_TOKEN,
            TMP_PAUSE_TOKEN,
            SYSTEM_BOS_TOKEN,
            USER_BOS_TOKEN,
            ASSISTANT_BOS_TOKEN,
            VOICE_ASSISTANT_BOS_TOKEN,
            IMAGE_PAD_TOKEN,
            DEFAULT_IMAGE_TOKEN,
        ]

        # Create a bidirectional mapping between special tokens and mask token.
        if len(self.extended_special_tokens_list) > 0:
            NUM_SPECIAL_TOKENS = len(self.extended_special_tokens_list)
            MASK_TOKEN_LIST = [
                f"<mask_{idx}>"
                for idx in range(MASK_START_IDX, MASK_START_IDX + NUM_SPECIAL_TOKENS)
            ]

            self.special_tokens_to_mask_tokens_map = {
                special_token: mask_token
                for special_token, mask_token in zip(
                    self.extended_special_tokens_list, MASK_TOKEN_LIST
                )
            }
            self.mask_tokens_to_special_tokens_map = {
                mask_token: special_token
                for special_token, mask_token in zip(
                    self.extended_special_tokens_list, MASK_TOKEN_LIST
                )
            }

            print(
                f"> Special tags are added by replacing the reserved mask labels in the vocabulary. The mapping between mask and special tags is as follows:"
            )
            print(
                ">\t Conversion list: \n\t"
                + "\n\t".join(
                    [
                        f"{m}: {s} (id: {self.convert_tokens_to_ids(m)})"
                        for m, s in self.mask_tokens_to_special_tokens_map.items()
                    ]
                )
            )

        self.extended_special_tokens_ids = self.convert_tokens_to_ids(
            self.extended_special_tokens_list
        )

        self.audio_pad_token = AUDIO_PAD_TOKEN
        self.audio_pad_token_id = self.convert_tokens_to_ids([self.audio_pad_token])[0]
        self.audio_bos_token = AUDIO_BOS_TOKEN
        self.audio_eos_token = AUDIO_EOS_TOKEN

        self.image_pad_token = IMAGE_PAD_TOKEN
        self.image_token = DEFAULT_IMAGE_TOKEN
        self.image_pad_token_id = self.convert_tokens_to_ids([self.image_pad_token])[0]

        self.conversation_bos_token = kwargs.get(
            "conversation_bos_token", self.bos_token
        )
        self.conversation_eos_token = kwargs.get(
            "conversation_eos_token", self.eos_token
        )
        self.conversation_eos_id = self.convert_tokens_to_ids(
            [self.conversation_eos_token]
        )[0]
        self.tmp_pause_token = kwargs.get("tmp_pause_token", TMP_PAUSE_TOKEN)
        self.use_text_instruction = kwargs.get("use_text_instruction", True)
        self.add_round_idx = kwargs.get("add_round_idx", True)
        self.simplify_conversation_format = kwargs.get(
            "simplify_conversation_format", False
        )
        self.auto_insert_system_prompt = kwargs.get("auto_insert_system_prompt", False)

        self.audio_head_num = kwargs.get(
            "audio_head_num", self.config["audio_head_num"]
        )

        print(f"> [Tokenizer] conversation_bos_token: {self.conversation_bos_token}")
        print(f"> [Tokenizer] conversation_eos_token: {self.conversation_eos_token}")
        print(f"> [Tokenizer] tmp_pause_token: {self.tmp_pause_token}")
        print(f"> [Tokenizer] use_text_instruction: {self.use_text_instruction}")
        print(f"> [Tokenizer] add_round_idx: {self.add_round_idx}")
        print(
            f"> [Tokenizer] simplify_conversation_format: {self.simplify_conversation_format}"
        )
        print(
            f"> [Tokenizer] auto_insert_system_prompt: {self.auto_insert_system_prompt}"
        )

        self.extended_special_tokens_start_idx = min(self.extended_special_tokens_ids)
        self.pause_id = self.convert_tokens_to_ids([self.tmp_pause_token])[0]
        print(f"> [Tokenizer] pause_id: {self.pause_id}")

        kwargs.update({"conversation_eos_token": self.conversation_eos_token})
        kwargs.update({"tokenizer": self.tokenizer})
        self.text_converter = TextSftDataConverter(**kwargs)
        self.audio_converter = AudioSftDataConverter(**kwargs)
        self.audio_codec_converter = AudioCodecSftDataConverter(**kwargs)
        self.image_converter = ImageSftDataConverter(**kwargs)
        self.multimodal_interleaved_converter = MultimodalInterleavedConverter(**kwargs)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def special_tokens_map(self):
        return {
            token: token_idx
            for token, token_idx in zip(
                self.extended_special_tokens_list, self.extended_special_tokens_ids
            )
        }

    def __call__(self, *kargs, **kwargs):
        return self.tokenizer(*kargs, **kwargs)

    def convert_ids_to_tokens(self, token_ids, *kargs, **kwargs):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids, *kargs, **kwargs)
        return [
            (
                self.mask_tokens_to_special_tokens_map[t]
                if t in self.mask_tokens_to_ids_map
                else t
            )
            for t in tokens
        ]

    def convert_tokens_to_ids(self, tokens, *kargs, **kwargs):
        if len(self.extended_special_tokens_list) > 0:
            if isinstance(tokens, list):
                tokens = [
                    (
                        self.special_tokens_to_mask_tokens_map[token]
                        if token in self.extended_special_tokens_list
                        else token
                    )
                    for token in tokens
                ]
            else:
                assert isinstance(tokens, str)
                if tokens in self.extended_special_tokens_list:
                    tokens = self.special_tokens_to_mask_tokens_map[tokens]
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def __len__(self):
        return len(self.tokenizer)

    def _convert_tokens_based_map(self, str, tokens_maps):
        for ori_token, new_token in tokens_maps.items():
            str = str.replace(ori_token, new_token)
        return str

    def convert_extented_special_tokens_to_mask_tokens(self, str):
        return self._convert_tokens_based_map(
            str, self.special_tokens_to_mask_tokens_map
        )

    def convert_extented_mask_tokens_to_special_tokens(self, str):
        return self._convert_tokens_based_map(
            str, self.mask_tokens_to_special_tokens_map
        )

    def encode(self, text, *kargs, **kwargs):
        if len(self.extended_special_tokens_list) > 0:
            text = self.convert_extented_special_tokens_to_mask_tokens(text)
        return self.tokenizer.encode(text, *kargs, **kwargs)

    def decode(self, token_ids, *kargs, **kwargs):
        text = self.tokenizer.decode(token_ids, *kargs, **kwargs)
        if len(self.extended_special_tokens_list) > 0:
            text = self.convert_extented_mask_tokens_to_special_tokens(text)
        return text

    def batch_decode(self, *kargs, **kwargs):
        return self.tokenizer.batch_decode(*kargs, **kwargs)

    def _content_convert(self, role, content, prefix_handle_fn, suffix_handle_fn):
        content_input_ids, content_input_text, content_multi_codec_ids = [], [], []
        audios, images = [], []

        content_prefix = prefix_handle_fn()
        content_prefix_ids = self.encode(content_prefix)
        prefix_len = len(content_prefix_ids)

        if self.config["enable_audio"]:
            content_multi_codec_ids.extend(
                [[CODEC_PAD_ID] * self.audio_head_num] * prefix_len
            )

        for item in content:
            c_type = item["type"]

            converter = self.text_converter
            kwargs = {}
            if c_type == "audio":
                converter = self.audio_converter
            elif c_type == "audio_codec":
                converter = self.audio_codec_converter
                if role == "user":
                    kwargs["is_text_pad"] = True
            elif c_type == "image" or c_type == "video":
                converter = self.image_converter
            elif c_type == "multimodal_interleaved":
                converter = self.multimodal_interleaved_converter
            else:
                converter = self.text_converter

            output = converter.convert(item, self.encode, **kwargs)
            item_input_ids = output["input_ids"]
            item_input_text = output["input_text"]
            item_multi_codec_id = output.get("multi_codec_ids", None)
            if self.config["enable_audio"] and item_multi_codec_id is None:
                item_multi_codec_id = [[CODEC_PAD_ID] * self.audio_head_num] * len(
                    item_input_ids
                )

            if (
                role == "system"
                or role == "assistant"
                or role == "tool"
                or role == "function"
            ):
                item_input_text += self.conversation_eos_token
                item_input_ids += self.encode(self.conversation_eos_token)
                if self.config["enable_audio"]:
                    item_multi_codec_id.append([CODEC_PAD_ID] * self.audio_head_num)

            content_input_text.append(item_input_text)
            content_input_ids.extend(item_input_ids)
            if self.config["enable_audio"]:
                content_multi_codec_ids.extend(item_multi_codec_id)

            item_audios = output.get("audios", None)
            if item_audios is not None:
                audios.append(item_audios)
            item_images = output.get("images", None)
            if item_images is not None:
                images.append(item_images)

        if suffix_handle_fn:
            content_suffix = suffix_handle_fn()
            content_suffix_ids = self.encode(content_suffix)
            if self.config["enable_audio"]:
                content_multi_codec_ids.extend(
                    [[CODEC_PAD_ID] * self.audio_head_num] * len(content_suffix_ids)
                )
        else:
            content_suffix = ""
            content_suffix_ids = []

        input_text = [content_prefix] + content_input_text + [content_suffix]
        input_ids = content_prefix_ids + content_input_ids + content_suffix_ids

        return (
            input_ids,
            input_text,
            content_multi_codec_ids,
            audios,
            images,
            (len(content_prefix_ids), len(content_input_ids), len(content_suffix_ids)),
        )

    def apply_chat_template(self, messages, *kargs, **kwargs):
        include_audio = False
        for message in messages:
            if include_audio:
                break
            for item in message["content"]:
                if item["type"] == "audio":
                    include_audio = True
                    break

        is_voice_chat = kwargs.get("is_voice_chat", False)
        auto_insert_system_prompt = kwargs.get(
            "auto_insert_system_prompt", self.auto_insert_system_prompt
        )
        simplify_conversation_format = kwargs.get(
            "simplify_conversation_format", self.simplify_conversation_format
        )
        degenerate_to_textual_form = kwargs.get("degenerate_to_textual_form", False)
        infer_insert_assistant = kwargs.get("infer_insert_assistant", False)

        output_format = kwargs.get("output_format", "str")
        return_pt_tensors = kwargs.get("return_pt_tensors", False)

        input_ids, role_ids, multi_codec_ids = [], [], []

        conversation_format_list = []

        audios = []
        images = []

        for turn_idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if degenerate_to_textual_form and role == "voice assistant":
                role = "assistant"
                content = self.audio_codec_converter.degenerate_to_textual_form(content)

            # system prompt
            if turn_idx == 0 and role != "system" and auto_insert_system_prompt:
                if include_audio or is_voice_chat:
                    default_system_str = (
                        "You are an emotional voice intelligent assistant."
                    )
                else:
                    default_system_str = "You are a helpful assistant."

                if simplify_conversation_format:
                    system_content = f"{SYSTEM_BOS_TOKEN}{default_system_str}{self.conversation_eos_token}"
                else:
                    system_content = f"{self.conversation_bos_token}system\n{default_system_str}{self.conversation_eos_token}\n"
                conversation_format_list.append(system_content)

                system_ids = self.encode(system_content)
                role_ids.extend([ROLE_FLAG_IDS["system"]] * len(system_ids))
                input_ids.extend(system_ids)
                multi_codec_ids.extend(
                    [[CODEC_PAD_ID] * self.audio_head_num] * len(system_ids)
                )

            if "voice" in role:

                def voice_assistant_prefix():
                    if self.use_text_instruction:
                        content_prefix = VOICE_ASSISTANT_PREFIX
                    else:
                        if simplify_conversation_format:
                            voice_ids = re.findall(r"voice_(\d+)", role)
                            voice_role = (
                                "" if len(voice_ids) == 0 else f"<voice_{voice_ids[0]}>"
                            )
                            content_prefix = f"{VOICE_ASSISTANT_BOS_TOKEN}{voice_role}"
                        else:
                            content_prefix = f"{self.conversation_bos_token}{role}\n"
                    return content_prefix

                def voice_assistant_suffix():
                    content_suffix = "" if simplify_conversation_format else "\n"
                    return content_suffix

                (
                    content_input_ids,
                    content_input_text,
                    content_multi_codec_ids,
                    content_audios,
                    content_images,
                    (len_prefix_ids, len_body_ids, len_suffix_ids),
                ) = self._content_convert(
                    role=role,
                    content=content,
                    prefix_handle_fn=voice_assistant_prefix,
                    suffix_handle_fn=voice_assistant_suffix,
                )

                conversation_format_list.append("".join(content_input_text))
                input_ids.extend(content_input_ids)
                if self.config["enable_audio"]:
                    multi_codec_ids.extend(content_multi_codec_ids)
                pid, rid = ROLE_FLAG_IDS["pad"], ROLE_FLAG_IDS[role]
                role_ids.extend(
                    [pid] * len_prefix_ids
                    + [rid] * len_body_ids
                    + [pid] * len_suffix_ids
                )

            elif role == "user":
                content_input_ids, content_input_text, content_multi_codec_ids = (
                    [],
                    [],
                    [],
                )

                def user_prefix():
                    round_idx = turn_idx // 2
                    round_str = f"[Round {round_idx}] " if self.add_round_idx else ""
                    content_prefix = f"{round_str}"
                    if self.use_text_instruction:
                        content_prefix += f"{role.upper()}:"
                    else:
                        content_prefix += (
                            f"{USER_BOS_TOKEN}"
                            if simplify_conversation_format
                            else f"{self.conversation_bos_token}{role}\n"
                        )
                    return content_prefix

                def user_suffix():
                    content_suffix = "\n"
                    return content_suffix

                (
                    content_input_ids,
                    content_input_text,
                    content_multi_codec_ids,
                    content_audios,
                    content_images,
                    (len_prefix_ids, len_body_ids, len_suffix_ids),
                ) = self._content_convert(
                    role=role,
                    content=content,
                    prefix_handle_fn=user_prefix,
                    suffix_handle_fn=user_suffix,
                )
                if content_audios:  # list
                    audios.extend(content_audios)
                if content_images:  # list
                    images.extend(content_images)

                conversation_format_list.append("".join(content_input_text))
                input_ids.extend(content_input_ids)
                if self.config["enable_audio"]:
                    multi_codec_ids.extend(content_multi_codec_ids)
                role_ids.extend([ROLE_FLAG_IDS[role]] * len(content_input_ids))

            else:

                def else_prefix():  # system/assistant/tool
                    if self.use_text_instruction:
                        content_prefix = f"{role.upper()}:"
                    else:
                        if simplify_conversation_format:
                            bos_token = (
                                SYSTEM_BOS_TOKEN
                                if role == "system"
                                else (
                                    ASSISTANT_BOS_TOKEN
                                    if role == "assistant"
                                    else USER_BOS_TOKEN
                                )
                            )
                            content_prefix = f"{bos_token}"
                        else:
                            content_prefix = f"{self.conversation_bos_token}{role}\n"
                    return content_prefix

                def else_suffix():
                    content_suffix = "" if simplify_conversation_format else "\n"
                    return content_suffix

                (
                    content_input_ids,
                    content_input_text,
                    content_multi_codec_ids,
                    content_audios,
                    content_images,
                    (len_prefix_ids, len_body_ids, len_suffix_ids),
                ) = self._content_convert(
                    role=role,
                    content=content,
                    prefix_handle_fn=else_prefix,
                    suffix_handle_fn=else_suffix,
                )

                conversation_format_list.append("".join(content_input_text))
                input_ids.extend(content_input_ids)
                if self.config["enable_audio"]:
                    multi_codec_ids.extend(content_multi_codec_ids)
                pid, rid = ROLE_FLAG_IDS["pad"], ROLE_FLAG_IDS[role]
                role_ids.extend(
                    [pid] * len_prefix_ids
                    + [rid] * len_body_ids
                    + [pid] * len_suffix_ids
                )

        # Automatically insert the ASSISTANT: tag during the inference process.
        if infer_insert_assistant:

            def infer_assistant_prefix():  # system/assistant/tool
                if self.use_text_instruction:
                    content_prefix = (
                        "ASSISTANT:" if not is_voice_chat else VOICE_ASSISTANT_PREFIX
                    )
                else:
                    if simplify_conversation_format:
                        bos_token = (
                            ASSISTANT_BOS_TOKEN
                            if role == "assistant"
                            else USER_BOS_TOKEN
                        )
                        content_prefix = f"{bos_token}"
                    else:
                        content_prefix = f"{self.conversation_bos_token}{role}\n"
                return content_prefix

            (
                content_input_ids,
                content_input_text,
                content_multi_codec_ids,
                _,
                _,
                (len_prefix_ids, _, _),
            ) = self._content_convert(
                role="assistant",
                content=[],
                prefix_handle_fn=infer_assistant_prefix,
                suffix_handle_fn=None,
            )

            conversation_format_list.append("".join(content_input_text))
            input_ids.extend(content_input_ids)
            if self.config["enable_audio"]:
                multi_codec_ids.extend(content_multi_codec_ids)
            pid = ROLE_FLAG_IDS["pad"]
            role_ids.extend([pid] * len_prefix_ids)

        conversations = (
            conversation_format_list
            if output_format == "list"
            else "".join(conversation_format_list)
        )

        format_results = {"conversations": conversations}

        if return_pt_tensors:
            format_results["input_ids"] = torch.LongTensor(input_ids)
            format_results["role_ids"] = torch.LongTensor(role_ids)
            if self.config["enable_audio"]:
                format_results["multi_codec_ids"] = torch.tensor(multi_codec_ids)

        format_results["audios"] = audios
        format_results["images"] = images
        return format_results

    def conversation_batch_encode(self, batch_messages, **kwargs):
        seq_length = self.config["seq_length"]
        batch_input_ids, batch_role_ids, batch_multi_codec_ids = [], [], []

        audios, images = [], []
        accumulated_seq_length = 0
        total_samples = len(batch_messages)

        for batch_index, messages in enumerate(batch_messages):
            try:
                outputs = self.apply_chat_template(
                    messages,
                    output_format="list",
                    return_pt_tensors=True,
                    **kwargs,
                )
            except Exception as e:
                traceback.print_exc()
                outputs = self.apply_chat_template(
                    messages,
                    output_format="list",
                    return_pt_tensors=True,
                    degenerate_to_textual_form=True,
                    **kwargs,
                )
                logger.warning(
                    f"> Abnormal data detected; it will be degraded to plain text format. {messages}, {e}"
                )

            input_ids = outputs["input_ids"]
            sample_len = input_ids.shape[-1]
            if batch_index > 0 and accumulated_seq_length + sample_len > seq_length:
                break

            accumulated_seq_length += sample_len

            batch_input_ids.append(input_ids)
            batch_role_ids.append(outputs["role_ids"])
            if self.config["enable_audio"]:
                batch_multi_codec_ids.append(outputs["multi_codec_ids"])

            message_audios = outputs.get("audios", [])
            message_images = outputs.get("images", [])
            audios.extend(message_audios)
            images.extend(message_images)

        seq_lens = torch.LongTensor(
            [[input_ids.size(-1)] for input_ids in batch_input_ids]
        )
        max_len = max([input_ids.size(-1) for input_ids in batch_input_ids])
        padding_batch_input_ids, padding_batch_role_ids = [], []
        for input_ids, role_ids in zip(batch_input_ids, batch_role_ids):
            pad_len = max_len - input_ids.size(-1)
            padding_batch_input_ids.append(
                F.pad(input_ids, pad=(0, pad_len), value=self.pad_token_id)
            )
            padding_batch_role_ids.append(
                F.pad(role_ids, pad=(0, pad_len), value=PAD_FLAG_ID)
            )

        input_ids = torch.stack(padding_batch_input_ids, dim=0)
        role_ids = torch.stack(padding_batch_role_ids, dim=0)

        attention_mask = (
            torch.arange(max_len).unsqueeze(0).expand_as(input_ids) < seq_lens
        )

        data = {
            "input_ids": input_ids,
            "role_ids": role_ids,
            "attention_mask": attention_mask,
            "system_mask": role_ids == SYSTEM_FLAG_ID,
            "user_mask": role_ids == USER_FLAG_ID,
            "assistant_mask": role_ids == ASSISTANT_FLAG_ID,
            "audios": audios,
            "images": images,
        }

        if len(batch_multi_codec_ids) > 0:
            max_codec_len = max(
                [codec_ids.size(0) for codec_ids in batch_multi_codec_ids]
            )
            padding_batch_multi_codec_ids = []
            for codec_ids in batch_multi_codec_ids:
                pad_len = max_codec_len - codec_ids.size(0)
                padding_batch_multi_codec_ids.append(
                    F.pad(
                        codec_ids,
                        pad=(
                            (0, 0, pad_len, 0)
                            if self.padding_side == "left"
                            else (0, 0, 0, pad_len)
                        ),
                        value=self.pad_token_id,
                    )
                )

            data["multi_codec_ids"] = torch.stack(padding_batch_multi_codec_ids, dim=0)

        return data

    def save_vocabulary(self, *kargs, **kwargs):
        return self.tokenizer.save_vocabulary(*kargs, **kwargs)

    def save_pretrained(self, *kargs, **kwargs):
        return self.tokenizer.save_pretrained(*kargs, **kwargs)
