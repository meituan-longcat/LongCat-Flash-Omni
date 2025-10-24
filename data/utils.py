import math
import os
import random
import re
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal

import decord
import imageio
import librosa
import numpy as np
import torch
import torchaudio
import torchvision
from decord import VideoReader, cpu
from PIL import Image

from encoders.utils import LongCatImageTransform
from global_vars import get_config

MAX_SAMPLE_NUM_FRAME_PER_VIDEO = 8
AUDIO_INPUT_FRAME_SEC = 0.08


def cal_samplefps(duration_sec, max_duration):
    if duration_sec <= 8:
        return 0.5
    else:
        sample_fps = math.ceil(duration_sec / max_duration)
        return sample_fps


def round_decimal(x, round_index="1"):
    if round_index == "1":
        return int(Decimal(x).quantize(Decimal(round_index), rounding=ROUND_HALF_UP))
    else:
        return float(Decimal(x).quantize(Decimal(round_index), rounding=ROUND_HALF_UP))


def refix_frame_index(frame_idx, video_fps):
    sec_msg = []

    frame_idx_double = []
    for i in range(0, len(frame_idx), 2):

        pre_frame_idx = frame_idx[i]
        frame_idx_double.append(pre_frame_idx)
        frame_idx_double.append(frame_idx[i + 1])

        idx = frame_idx[i]
        sec_float = round_decimal(idx / video_fps, "0.1")
        sec_str = f"second{{{sec_float}}}:"
        sec_msg.append(sec_str)

    frame_idx = frame_idx_double

    return frame_idx, sec_msg


def get_index(
    video_fps,
    video_max_frame,
    bound=None,
    first_idx=0,
    sample_fps=1,
    frames_upbound=32,
    fps_upbound=4,
    frames_downbound=16,
):
    clip_bound = bound
    if clip_bound:
        start, end = clip_bound[0], clip_bound[1]
    else:
        start, end = -10000, 1000000

    start_idx = max(first_idx, round_decimal(float(start) * video_fps))
    end_idx = min(round_decimal(float(end) * video_fps), video_max_frame)

    if video_fps <= sample_fps:
        frame_idx = [
            math.floor(i) for i in np.arange(start_idx, end_idx, 1 / sample_fps)
        ]
    else:
        seg_size = (
            round_decimal(video_fps / sample_fps)
            if sample_fps > 0 and video_fps > 0
            else 1
        )
        frame_idx = [
            min(round_decimal(i), end_idx - 1)
            for i in np.arange(start_idx + seg_size / 2, end_idx, seg_size)
        ]
        frame_idx = [int(cur_idx) for cur_idx in frame_idx]

        if fps_upbound is not None:
            higher_fps = min(
                frames_downbound / ((end_idx - start_idx) / video_fps), fps_upbound
            )
            if higher_fps > sample_fps:
                higher_steps = round_decimal(video_fps / higher_fps)
                frame_idx = [
                    min(round_decimal(i), end_idx - 1)
                    for i in np.arange(
                        start_idx + higher_steps / 2, end_idx, higher_steps
                    )
                ]

        if frames_upbound > 0:
            if len(frame_idx) > frames_upbound:
                uniform_sampled_frames = np.linspace(
                    start_idx, end_idx - 1, frames_upbound, dtype=int
                )
                frame_idx = uniform_sampled_frames.tolist()

    frame_idx = sorted(list(frame_idx))
    if len(frame_idx) % 2 != 0:
        last_img = frame_idx[-1]
        frame_idx.append(last_img)

    if len(frame_idx) < 1:
        raise ValueError(
            f"frame_indices error,start{start}, end{end}, max_frame{video_max_frame}"
        )

    return refix_frame_index(frame_idx, video_fps)


def calculate_token_num_after_post_pooling(
    cur_sample_num, cur_sample_grid_shapes, ratio, config, after_adapter=False
):

    if ratio <= 1.0:
        return (
            [grid_shapes[1:] for grid_shapes in cur_sample_grid_shapes],
            cur_sample_num,
        )

    min_tokens_video = config.min_tokens_video
    max_tokens_video = config.max_tokens_video

    resized_shapes, resized_tokens = [], []
    for meta_idx, meta in enumerate(cur_sample_num):
        max_token_to_resize = min(
            int(cur_sample_num[meta_idx] / ratio), config.max_tokens
        )
        min_token_to_resize = min_tokens_video
        t, height, width = cur_sample_grid_shapes[meta_idx]
        if not after_adapter:
            width = width // 2
        resized_height, resized_width = LongCatImageTransform.smart_resize(
            height * config.patch_size,
            width * config.patch_size,
            2 * config.patch_size,
            min_token_to_resize * config.patch_size * config.patch_size,
            max_token_to_resize * config.patch_size * config.patch_size,
        )
        resized_height //= config.patch_size
        resized_width //= config.patch_size
        resized_tokens.append(resized_height * resized_width)
        if not after_adapter:
            resized_width *= 2
        resized_shapes.append([resized_height, resized_width])

    return resized_shapes, resized_tokens


def get_ar_token_num(
    univitar_config, grid_shapes, tokenizer=None, downsample: bool = True
):
    if not isinstance(grid_shapes[0], int):
        return [
            get_ar_token_num(univitar_config, x, tokenizer, downsample=downsample)
            for x in grid_shapes
        ]
    t, h, w = grid_shapes
    if not downsample:
        return t * h * w
    token_num = 0.5 * w * h
    return int(token_num)


def get_image_token_num(univitar_config, images_list, tokenizer=None):
    if len(images_list) == 0:
        return [], []
    _grid_shapes = [x.shape for x in images_list]
    _grid_shapes = [
        (
            x[0] // 2,
            x[2] // univitar_config.patch_size,
            x[3] // univitar_config.patch_size,
        )
        for x in _grid_shapes
    ]
    image_token_nums = get_ar_token_num(
        univitar_config=univitar_config,
        grid_shapes=_grid_shapes,
        tokenizer=tokenizer,
    )

    return image_token_nums, _grid_shapes


def unified_collate_fn_for_vlm():
    from torch.utils.data import default_collate

    def collate_fn(batch):
        special_collate_keys = ["image"]
        media_list = []
        for meta in batch:
            for media in special_collate_keys:
                media_list.extend(meta[media] if meta[media] is not None else [])

        special_collate_keys.extend(["resized_image_token_nums", "resized_grid_shapes"])
        resized_image_token_nums, resized_grid_shapes, is_null_img_flag = [], [], False
        for meta in batch:
            if meta.get("num_image") == 0:
                is_null_img_flag = True
            resized_image_token_nums.extend(meta.get("resized_image_token_nums", []))
            resized_grid_shapes.extend(meta.get("resized_grid_shapes", []))

        c, temporal_patch_size, spatial_patch_size, spatial_merge_size = 3, 2, 14, 1
        one_grid_dim = (
            c * temporal_patch_size * spatial_patch_size * spatial_patch_size
        )  # 3 * 2 * 14 * 14
        patches_list, grid_list = [], []
        for one_data in media_list:
            t, c, h, w = one_data.shape
            grid_t = t // temporal_patch_size
            grid_h, grid_w = h // spatial_patch_size, w // spatial_patch_size
            one_data = one_data.reshape(
                grid_t,
                temporal_patch_size,
                c,
                grid_h // spatial_merge_size,
                spatial_merge_size,
                spatial_patch_size,
                grid_w // spatial_merge_size,
                spatial_merge_size,
                spatial_patch_size,
            )
            one_data = one_data.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = one_data.reshape(
                grid_t * grid_h * grid_w, one_grid_dim
            ).contiguous()
            patches_list.append(flatten_patches)
            grid_list.append((grid_t, grid_h, grid_w))

        grid_list = torch.tensor(grid_list)
        if is_null_img_flag:
            resized_image_token_nums = [grid_h * grid_w // 2]
            resized_grid_shapes = [[grid_h, grid_w]]
        ret = {
            "image": torch.concatenate(patches_list, dim=0),
            "grid_shapes": grid_list,
            "resized_image_token_nums": resized_image_token_nums,
            "resized_grid_shapes": resized_grid_shapes,
        }

        default_collate_keys = [
            x for x in batch[0].keys() if x not in special_collate_keys
        ]
        default_collated_items = {
            key: default_collate([meta[key] for meta in batch])
            for key in default_collate_keys
        }
        ret = ret | default_collated_items

        return ret

    return collate_fn


# @timeout_decorator.timeout(10)
def __load_audio(audio_path, sample_rate):

    waveform, sr = librosa.load(audio_path, sr=sample_rate, mono=True, dtype="float32")
    waveform = torch.from_numpy(waveform).reshape(1, -1)

    # resample
    if sr != sample_rate:
        waveform = librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=sample_rate)
        waveform = torch.from_numpy(waveform).reshape(1, -1)

    # to mono
    waveform = waveform.mean(dim=0) if waveform.size(0) != 1 else waveform.squeeze(0)

    return waveform


def _load_audio(audio_file, sample_rate):
    try:
        return __load_audio(audio_file, sample_rate)
    except Exception as e:
        raise Exception(f"Failed to load audio file {audio_file} or timed out, {e}")


def load_audios(audio_files, sample_rate):
    config = get_config()

    if not isinstance(audio_files, list):
        audio_files = [audio_files]

    waveforms = [_load_audio(audio_file, sample_rate) for audio_file in audio_files]

    if isinstance(waveforms, list):
        waveforms = torch.cat(waveforms, dim=-1) if len(waveforms) > 1 else waveforms[0]

    wav_len = waveforms.size(-1)
    if wav_len > config["max_audio_duration"] * sample_rate:
        bos = random.randint(0, wav_len - config["max_audio_duration"] * sample_rate)
        eos = bos + config["max_audio_duration"] * sample_rate
        waveforms = waveforms[bos:eos]
    return waveforms


def _load_audio_data(message_item):
    c_type = message_item["type"]
    item_main = message_item[c_type]
    meta = message_item.get("meta", {})

    audio_file = item_main
    sample_rate = meta.get("sample_rate", 16000)
    overlap_info = None
    clip_info = None
    noise_audio_file = None
    waveforms = load_audios(
        audio_file,
        sample_rate,
    )

    message_item[c_type] = waveforms
    return message_item


def load_data(message_item):
    c_type = message_item["type"]

    if c_type == "audio":
        return _load_audio_data(message_item)

    elif c_type == "image":
        return _load_image_data(message_item)

    elif c_type == "video":
        return _load_video_data(message_item)

    elif c_type == "multimodal_interleaved":
        return _load_multimodal_interleave_data(message_item)

    return message_item


def _load_image_data(message_item):
    c_type = message_item["type"]
    assert c_type == "image"

    item = message_item.pop(c_type)
    img_pil = Image.open(item).convert("RGB")

    message_item[c_type] = img_pil

    image_token_num = message_item.get("image_token_num")
    if image_token_num is not None:
        message_item["meta"]["image_token_num"] = image_token_num
    return message_item


def extract_timestamp(text):
    pattern = r"second\{([0-9]+\.?[0-9]*)\}:"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def extract_and_update_sec_msgs(msgs, time_offset=0):
    secs = [extract_timestamp(sec) for sec in msgs]
    duration = max(secs)
    if time_offset > 0:
        msgs = [f"second{{{sec+time_offset:.1f}}}:" for sec in secs]
    return msgs, duration


def _load_video_data(message_item):
    config = get_config()

    assert isinstance(
        message_item, dict
    ), f"message_item 应该是 dict 格式，{type(message_item)}"
    c_type = message_item["type"]
    assert c_type == "video"

    video_path = message_item.pop(c_type)
    meta = message_item.get("meta", {})

    bound = meta["bound"] if "bound" in meta else None
    first_idx = meta["first_idx"] if "first_idx" in meta else 0
    prefetch_duration = (
        meta["prefetch_duration"] if "prefetch_duration" in meta else None
    )
    prefetch_fps = meta["fps"] if "fps" in meta else None
    prefetch_videoframe_max_frame = (
        meta["videoframe_max_frame"] if "videoframe_max_frame" in meta else None
    )
    frame_indices = meta.get("frame_indices", None)
    if prefetch_videoframe_max_frame:
        prefetch_videoframe_max_frame = int(prefetch_videoframe_max_frame)

    time_offset = meta.get("time_offset", 0.0)

    read_fps = config["read_fps"]
    sample_mid = config["sample_mid"]
    max_duration = config["max_frame_num"] * 2
    frame_num = config["frame_num"]

    img_pils, sec_msg = read_video_decord(
        video_path,
        bound=bound,
        frame_num=frame_num,
        abandon_size=5,
        max_duration=max_duration,
        first_idx=first_idx,
        read_fps=read_fps,
        sample_mid=sample_mid,
        prefetch_duration=prefetch_duration,
        prefetch_fps=prefetch_fps,
        frame_indices=frame_indices,
    )

    sec_msg, duration = extract_and_update_sec_msgs(sec_msg, time_offset)

    if len(img_pils) % 2 != 0:
        if meta["image_num"] * 2 > len(img_pils):
            img_pils.append(img_pils[-1])
        else:
            img_pils = img_pils[:-1]

    message_item[c_type] = img_pils
    message_item["meta"]["frame_secs"] = sec_msg
    message_item["meta"]["duration"] = duration
    return message_item


def _load_multimodal_interleave_data(message_item):
    assert (
        "video" in message_item["multimodal_interleaved"]
    ), f"Require that the message_item -> multimodal_interleaved contains video field."

    video_file = message_item["multimodal_interleaved"].get("video")
    meta_info = message_item["meta"]
    use_audio_track = meta_info.get("use_audio_track", False)

    assert (
        use_audio_track or "audio" in message_item["multimodal_interleaved"]
    ), f"Require that the message_item -> multimodal_interleaved contains audio field."

    frame_sample_strategy = meta_info.get("frame_sample_strategy", "sequential")
    average_sample_num_frame_per_video = meta_info.get(
        "average_sample_num_frame_per_video", 8
    )
    sequential_sample_num_frame_per_sec = meta_info.get(
        "sequential_sample_num_frame_per_sec", 1
    )
    sequential_max_num_sample_frame = meta_info.get(
        "sequential_max_num_sample_frame", -1
    )
    time_offset = meta_info.get("time_offset", 0.0)

    bound = meta_info.get("bound", None)
    frame_indices = meta_info.get("frame_indices", None)

    (
        imgs,
        secs,
        audio_track_waveform,
        duration,
        update_kwargs,
    ) = _load_and_sample_video_track(
        video_file,
        bound,
        time_offset,
        use_audio_track,
        frame_sample_strategy,
        average_sample_num_frame_per_video,
        sequential_sample_num_frame_per_sec,
        sequential_max_num_sample_frame,
        frame_indices,
    )
    frame_sample_strategy = update_kwargs.get("frame_sample_strategy")
    average_sample_num_frame_per_video = update_kwargs.get(
        "average_sample_num_frame_per_video"
    )

    # Read audio
    audio_file = message_item["multimodal_interleaved"].pop("audio", None)

    if audio_file is not None:
        waveform = _load_audio(audio_file, sample_rate=16000)
    else:
        waveform = None

    # Concatenate video audio track with audio
    if audio_track_waveform is not None:
        waveform = (
            torch.cat([audio_track_waveform, waveform], dim=0)
            if waveform is not None
            else audio_track_waveform
        )

    message_item["multimodal_interleaved"]["audio"] = waveform
    message_item["multimodal_interleaved"]["video"] = imgs
    message_item["meta"]["frame_secs"] = secs
    message_item["meta"]["frame_sample_strategy"] = frame_sample_strategy
    message_item["meta"][
        "average_sample_num_frame_per_video"
    ] = average_sample_num_frame_per_video
    message_item["meta"]["duration"] = duration

    return message_item


@contextmanager
def safe_videoreader(path, ctx=cpu(0), num_threads=1):
    vr = None
    try:
        vr = VideoReader(path, ctx=ctx, num_threads=num_threads)
        yield vr
    finally:
        if vr is not None:
            try:
                del vr
            except Exception:
                pass


def read_video_decord(
    video_path,
    bound=None,
    frame_num=16,
    abandon_size=5,
    max_duration=100,
    first_idx=0,
    read_fps=True,
    sample_mid=True,
    prefetch_duration=None,
    prefetch_fps=None,
    frame_indices=None,
):
    with safe_videoreader(video_path, ctx=cpu(0), num_threads=1) as vr:
        total_frame_num = len(vr)
        max_frame = total_frame_num - 1
        fps = float(vr.get_avg_fps())  ## video fps

        if prefetch_duration is not None and prefetch_fps is not None:
            max_frame = float(int(prefetch_duration * prefetch_fps) - 1)

        if bound is None:
            sample_fps = cal_samplefps(max_frame / fps, max_duration)
        else:
            sample_fps = cal_samplefps(bound[1] - bound[0], max_duration)

        if frame_indices:
            frame_id_list, sec_msg = frame_indices, None
        else:
            frame_id_list, sec_msg = get_index(
                fps,
                max_frame,
                bound=bound,
                first_idx=first_idx,
                sample_fps=sample_fps,
                frames_upbound=max_duration,
            )

        frame_id_list = [
            min(frame_index, max_frame - abandon_size) for frame_index in frame_id_list
        ]

        video = vr.get_batch(frame_id_list).asnumpy()  ##return numpy()
        video = [Image.fromarray(frame) for frame in video]
        # https://github.com/dmlc/decord/issues/208
        vr.seek(0)

    return video, sec_msg


def _load_and_sample_video_track(
    video_file,
    bound,
    time_offset,
    use_audio_track,
    frame_sample_strategy,
    average_sample_num_frame_per_video,
    sequential_sample_num_frame_per_sec,
    sequential_max_num_sample_frame,
    frame_indices,
):
    config = get_config()

    if frame_indices is not None:
        frame_sample_strategy = "average"
        average_sample_num_frame_per_video = len(frame_indices) // 2

    video_reader = None
    try:
        if use_audio_track:
            video_track, audio_track, fps_info = torchvision.io.read_video(
                filename=video_file, output_format="THWC"
            )
            audio_track_waveform = _audio_track_process(
                audio_track,
                sample_rate=fps_info.get("audio_fps"),
                target_sample_rate=16000,
            )
            total_video_frames, video_fps = (
                video_track.shape[0],
                fps_info.get("video_fps"),
            )
        else:
            audio_track_waveform = None
            video_reader = decord.VideoReader(
                video_file, ctx=decord.cpu(0), num_threads=1
            )
            total_video_frames, video_fps = (
                len(video_reader),
                video_reader.get_avg_fps(),
            )

        if bound is not None:
            start_pts, end_pts = bound[0], bound[1]
        else:
            start_pts, end_pts = None, None

        start_frame, end_frame, total_frames = calculate_video_frame_range(
            start_pts,
            end_pts,
            total_video_frames,
            video_fps,
        )
        frame_durations = [
            f"{i / video_fps + time_offset:.1f}" for i in range(total_video_frames + 1)
        ]

        duration = math.ceil(total_frames / video_fps)
        sample_num_frame = math.ceil(sequential_sample_num_frame_per_sec * duration)
        max_frame = (
            sample_num_frame // sequential_sample_num_frame_per_sec * video_fps
            + start_frame
        )
        sample_num_frame = sample_num_frame * 2
        sequential_max_num_sample_frame = sequential_max_num_sample_frame * 2
        if (
            sequential_max_num_sample_frame > 0
            and sample_num_frame > sequential_max_num_sample_frame
        ):
            frame_offset = 0  # Sampling frame offset, try to avoid sampling from the first and last frame.
            idx = torch.linspace(
                start_frame + frame_offset,
                end_frame - frame_offset,
                sequential_max_num_sample_frame,
            ).round()
            frame_sample_strategy = "average"
            average_sample_num_frame_per_video = sequential_max_num_sample_frame // 2
        else:
            frame_offset = 0
            idx = torch.linspace(
                start_frame + frame_offset,
                max_frame - frame_offset,
                sample_num_frame,
            ).round()
        idx[idx > end_frame] = end_frame
        idx = idx.long()

        if use_audio_track:
            video = video_track[idx].numpy()
        else:
            video = video_reader.get_batch(idx.tolist()).asnumpy()
            video_reader.seek(0)

        secs = [f"second{{{frame_durations[frame_idx]}}}:" for frame_idx in idx[::2]]

        imgs = [Image.fromarray(video[i]) for i in range(video.shape[0])]

    except Exception as e:
        print(
            f"An exception occurred during data processing in _load_and_sample_video_track:: {e}"
        )
        raise e
    finally:
        if not use_audio_track and video_reader is not None:
            del video_reader

    update_kwargs = {
        "frame_sample_strategy": frame_sample_strategy,
        "average_sample_num_frame_per_video": average_sample_num_frame_per_video,
    }

    return imgs, secs, audio_track_waveform, duration, update_kwargs


def _audio_track_process(waveform, sample_rate, target_sample_rate=16000):

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=target_sample_rate
        )

    waveform = waveform.mean(dim=0) if waveform.size(0) != 1 else waveform.squeeze(0)
    return waveform


def calculate_video_frame_range(
    video_start,
    video_end,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    return start_frame, end_frame, end_frame - start_frame + 1
