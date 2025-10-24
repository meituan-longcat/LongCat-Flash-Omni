import argparse
import asyncio
import json
import os
import os.path as osp
import signal
import sys

sys.path.insert(0, os.path.abspath("./longcat_audio_codec"))

import sglang as sgl
import torch
import torchaudio

from data.data_processor import DataProcessor
from encoders.audio_adaptor import LongCatOmniAudioAdaptor
from encoders.embedding import AudioEmbedding, TextEmbedding
from encoders.utils import load_config, v_post_squeeze_image_embedding
from encoders.vision_adaptor import LongCatOmniVisionAdaptor
from global_vars import get_config, get_tokenizer, set_global_variables
from post_process.unified_post_processor import OmniUnifiedPostProcessor


def init_global_config(args):
    model_path = args.model_path
    config = dict(
        model_path=model_path,
        text_vocab_size=131072,
        audio_vocab_size=8224,
        audio_head_num=4,
        token_type_pause=1,
        token_type_end=2,
        token_type_audio=3,
        token_type_text=4,
        token_type_audio_text=5,
        token_type_asr=6,
        token_asr=102,
        token_pause=101,
        padding_idx=3,
        seq_length=34816,
        has_sec_sep=True,
        use_text_instruction=True,
        add_round_idx=False,
        # audio
        enable_audio=True,
        max_audio_duration=120,
        # vison
        video_vit=True,
        max_frame_num=128,
        frame_num=8,
        read_fps=True,
        sample_mid=True,
        v_post_squeeze=True,
        vision_encoder_type="univitar",
        max_vision_length=28672,
        omni_video_fix_post_pooling_times=-1,
        omni_video_fix_post_pooling_tokens=-1,
        is_voice_chat=True,
        omni_streaming_sequential_interval=1,
        preprocessor_config=osp.join(model_path, "audio", "preprocessor_config.json"),
        vision_config=osp.join(model_path, "vision", "config.json"),
        tokenizer_name_or_path=model_path,
    )
    set_global_variables(config)


def create_sglang_engine(args):
    config = get_config()
    model_path = args.model_path
    sglang_extra_config = dict(
        architectures=["LongcatFlashOmniForCausalLM"],
        onmi_extra_info=dict(
            hf_path=model_path,
            num_multi_ids=config["audio_head_num"],
            audio_head_num=config["audio_head_num"],
            audio_vocab_size=config["audio_vocab_size"],
            audio_id_offset=32,
            audio_rep_penalty_window=30,
            text_rep_penalty_window=30,
            audio_repetition_penalty=1.1,
            has_proj=False,
            audio_embed_pt=osp.join(model_path, "audio", "audio_embeddings.pt"),
            audio_output_layer_pt=osp.join(
                model_path, "audio", "audio_output_layers.pt"
            ),
        ),
    )
    init_kwargs = dict(
        model_path=args.model_path,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        trust_remote_code=True,
        cuda_graph_max_bs=16,
        mem_fraction_static=0.7,
        json_model_override_args=json.dumps(sglang_extra_config),
    )
    if args.nodes is not None:
        assert args.nodes >= 1
        init_kwargs["nnodes"] = args.nodes
        init_kwargs["node_rank"] = args.node_rank
        init_kwargs["dist_init_addr"] = args.dist_init_addr
    engine = sgl.Engine(**init_kwargs)
    return engine


def load_ckpt(model: torch.nn.Module, path: str):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt)


class LoncatOmniInfer:
    def __init__(self, args):
        self.node_rank = args.node_rank
        self.config = get_config()
        self.device = torch.device(args.encoder_device)
        if self.node_rank == 0:
            self.build_modality_models()

        self.sglang_engine = create_sglang_engine(args)

        self.loop = asyncio.new_event_loop()
        self._request_id = 0
        print("init_done")

    def build_modality_models(
        self,
    ):
        model_path = self.config["model_path"]
        llm_config = load_config(f"{model_path}/config.json")
        self.text_embedding = TextEmbedding(
            llm_config["vocab_size"], llm_config["hidden_size"]
        )
        self.text_embedding.bfloat16()
        self.text_embedding.to(self.device)

        self._vision_pad_token_id = get_tokenizer().image_pad_token_id
        self.vision_adaptor_model = LongCatOmniVisionAdaptor(
            f"{model_path}/vision/config.json", self.text_embedding.embedding_dim
        )
        self.vision_adaptor_model.bfloat16()
        self.vision_adaptor_model.to(self.device)

        self._audio_pad_token_id = get_tokenizer().audio_pad_token_id
        self.audio_adaptor_model = LongCatOmniAudioAdaptor(
            f"{model_path}/audio/config.json", self.text_embedding.embedding_dim
        )
        self.audio_adaptor_model.bfloat16()
        self.audio_adaptor_model.to(self.device)

        self._padding_idx = 3
        self._audio_head_num = 4
        self.audio_embedding = AudioEmbedding(
            audio_vocab_size=8224,
            audio_head_num=self._audio_head_num,
            hidden_size=self.text_embedding.embedding_dim,
            padding_idx=self._padding_idx,
        )
        self.audio_embedding.bfloat16()
        self.audio_embedding.to(self.device)

        self._input_processor = DataProcessor(sample_rate=16000)
        self._postprocessor = OmniUnifiedPostProcessor(
            config_path=f"{model_path}/audio_codec/config.yaml",
            device=self.device,
            output_modality=["text", "audio"],
        )

        self.load_encoder_ckpt(model_path)

    def load_encoder_ckpt(self, model_path: str):
        ckpt = torch.load(
            f"{model_path}/language_model_embedding.pt", map_location="cpu"
        )
        self.text_embedding.load_state_dict(ckpt)
        ckpt = torch.load(f"{model_path}/vision/vision_model.pt", map_location="cpu")
        self.vision_adaptor_model.vision_encoder.load_state_dict(ckpt)
        ckpt = torch.load(
            f"{model_path}/vision/vision_projector.pt", map_location="cpu"
        )
        self.vision_adaptor_model.vision_projector.load_state_dict(ckpt)
        ckpt = torch.load(f"{model_path}/audio/audio_encoder.pt", map_location="cpu")
        self.audio_adaptor_model.audio_encoder.load_state_dict(ckpt)
        ckpt = torch.load(f"{model_path}/audio/audio_projector.pt", map_location="cpu")
        self.audio_adaptor_model.audio_projector.load_state_dict(ckpt)
        ckpt = torch.load(f"{model_path}/audio/audio_embeddings.pt", map_location="cpu")
        self.audio_embedding.load_state_dict(ckpt)

    def _process_input(self, input_dict):
        data = self._input_processor.process(input_dict)
        prompts = data["prompts"]
        assert prompts.shape[0] == 1
        input_ids = prompts[:, :, 0]
        codecs = prompts[:, :, 1:]
        result = dict(
            input_ids=input_ids,
            codecs=codecs,
            images=data["images"],
            grid_shapes=data["grid_shapes"],
            resized_image_token_nums=data["resized_image_token_nums"],
            resized_grid_shapes=data["resized_grid_shapes"],
            audios=data["audios"],
            audio_masks=data["audio_masks"],
        )
        return result

    @torch.no_grad()
    def _get_input_embedding(
        self,
        input_ids,
        images=None,
        grid_shapes=None,
        resized_image_token_nums=None,
        resized_grid_shapes=None,
        codecs=None,
        audios=None,
        audio_masks=None,
    ):
        device = self.device
        input_ids = input_ids.to(device)
        merged_embeddings = self.text_embedding(input_ids)
        if codecs is not None:
            codecs = codecs.to(device)
            padding_mask = input_ids == self._padding_idx
            merged_embeddings[padding_mask] = 0.0
            audio_embeddings, audio_padding_mask = self.audio_embedding(codecs)
            for i in range(self._audio_head_num):
                merged_embeddings = merged_embeddings + audio_embeddings[i]
            padding_mask = torch.logical_and(padding_mask, audio_padding_mask)
            padding_idx_embedding = self.text_embedding(
                torch.tensor(self._padding_idx).to(merged_embeddings.device),
            )
            merged_embeddings[padding_mask] = padding_idx_embedding.expand(
                merged_embeddings[padding_mask].shape
            )

        if audios is not None:
            audios = audios.bfloat16().to(device)
            audio_masks = audio_masks.to(device)
            audio_adaptor_embedding = self.audio_adaptor_model(audios, audio_masks)
            audio_fill_mask = input_ids == self._audio_pad_token_id
            merged_embeddings[audio_fill_mask] = audio_adaptor_embedding

        if images is not None:
            images = images.bfloat16().to(device)
            if isinstance(grid_shapes, torch.Tensor):
                grid_shapes = grid_shapes.tolist()
            vision_embedding = self.vision_adaptor_model(images, grid_shapes)
            vision_embedding = v_post_squeeze_image_embedding(
                vision_embedding,
                grid_shapes,
                resized_image_token_nums,
                resized_grid_shapes,
            )
            vision_fill_mask = input_ids == self._vision_pad_token_id
            merged_embeddings[vision_fill_mask] = vision_embedding
        assert merged_embeddings.shape[0] == 1
        merged_embeddings = merged_embeddings[0]
        return merged_embeddings

    def generate(self, input, sampling_params=None):
        if self.node_rank != 0:
            return
        processed_data = self._process_input(input)
        input_embedding = self._get_input_embedding(**processed_data)
        input_embedding = input_embedding.cpu().tolist()
        self._request_id += 1

        async def generator():
            output = await self.sglang_engine.async_generate(
                input_embeds=input_embedding,
                sampling_params=sampling_params,
                stream=False,
            )
            result = self._postprocessor.process(output, tokenizer=get_tokenizer())
            return result

        result = self.loop.run_until_complete(generator())
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="The Longcat-omni model path."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="The output dir."
    )
    parser.add_argument("--nodes", type=int, default=None, help="The number of nodes.")
    parser.add_argument("--node-rank", type=int, default=0, help="The node rank")
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        default=None,
        help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`)",
    )
    parser.add_argument(
        "--tp-size", type=int, default=8, help="The tensor parallel size."
    )
    parser.add_argument(
        "--ep-size", type=int, default=8, help="The expert parallel size."
    )
    parser.add_argument(
        "--encoder-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="The placement device of the modailty encoder",
    )
    args = parser.parse_args()
    init_global_config(args)

    os.makedirs(args.output_dir, exist_ok=True)
    infer_engine = LoncatOmniInfer(args)
    sampling_params = {
        "temperature": 1,
        "max_new_tokens": 4096,
        "top_p": 1,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
    }
    from examples_dict import get_test_case_dict

    cases = get_test_case_dict()
    for case_name, input in cases.items():
        result = infer_engine.generate(input, sampling_params=sampling_params)
        if result is not None and result.audio_waveform is not None:
            wav_file = osp.join(args.output_dir, f"{case_name}.wav")
            torchaudio.save(
                wav_file,
                result.audio_waveform.cpu().squeeze(0),
                sample_rate=infer_engine._postprocessor.codec_processor.decoder.output_rate,
            )
            output_json = osp.join(args.output_dir, f"{case_name}.json")
            with open(output_json, "w", encoding="utf-8") as file:
                json.dump(
                    {"text": result.text, "audio_codes": result.audio_codec_tokens},
                    file,
                    ensure_ascii=False,
                    indent=4,
                )
        if result is not None:
            print(result)

    if args.node_rank != 0:
        signal.pause()


if __name__ == "__main__":
    main()
