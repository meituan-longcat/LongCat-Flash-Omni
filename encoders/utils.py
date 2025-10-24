import json
import math
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from PIL import Image
from torchvision import transforms


def load_config(config_file: str, **kwargs) -> dict:
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    for key in kwargs:
        config_dict[key] = kwargs[key]
    return config_dict


def get_ar_token_num(grid_shapes, downsample: bool = True):
    if not isinstance(grid_shapes[0], int):
        return [get_ar_token_num(x, downsample=downsample) for x in grid_shapes]
    t, h, w = grid_shapes
    if not downsample:
        return t * h * w
    token_num = 0.5 * w * h
    return int(token_num)


def get_2d_pool(image_feature, ori_shape, scaled_shape):
    num_frames, height, width = ori_shape
    if height == scaled_shape[0] and width == scaled_shape[1]:
        return image_feature
    image_feature = image_feature.view(num_frames, height, width, -1).contiguous()
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
    image_feature = torch.nn.functional.interpolate(
        image_feature, size=scaled_shape, mode="bilinear"
    )
    image_feature = image_feature.permute(0, 2, 3, 1).contiguous()
    num_dim = image_feature.shape[-1]
    image_feature = image_feature.view(-1, num_dim).contiguous()
    return image_feature


def update_vision_sequence_index_info(image_token_idx, cum_image_token_num):
    image_token_idx += 1
    idx_start, idx_end = (
        cum_image_token_num[image_token_idx],
        cum_image_token_num[image_token_idx + 1],
    )
    image_token_num = idx_end - idx_start
    slice_idx = slice(idx_start, idx_end)
    return image_token_idx, image_token_num, slice_idx


def v_post_squeeze_image_embedding(
    image_embedding, grid_shapes, resized_image_token_nums, resized_grid_shapes
):
    grid_shapes = deepcopy(grid_shapes)
    resized_grid_shapes = deepcopy(resized_grid_shapes)
    assert image_embedding.ndim == 2
    image_token_num_list = get_ar_token_num(grid_shapes=grid_shapes)
    assert len(image_embedding) == sum(
        image_token_num_list
    ), f"wrong image_token num, {len(image_embedding)} != {sum(image_token_num_list)}"

    cum_image_token_num = np.cumsum([0] + image_token_num_list)

    image_token_idx = -1
    image_embedding_pool = []

    for _ in image_token_num_list:
        image_token_idx, _, slice_idx_ = update_vision_sequence_index_info(
            image_token_idx, cum_image_token_num
        )
        cur_image_embedding = image_embedding[slice_idx_]

        resized_image_token_num = resized_image_token_nums[image_token_idx]
        scaled_shape = resized_grid_shapes[image_token_idx]
        ori_shape = grid_shapes[image_token_idx]
        # grid_shape needs to be downsampled
        ori_shape[-1] //= 2
        scaled_shape[-1] //= 2
        if ori_shape[1:] != scaled_shape:
            cur_image_embedding = get_2d_pool(
                cur_image_embedding, ori_shape, scaled_shape
            )
            assert resized_image_token_num == cur_image_embedding.shape[0]
        image_embedding_pool.append(cur_image_embedding)
    image_embedding = torch.concatenate(image_embedding_pool, axis=0)
    return image_embedding


class LongCatImageTransform(object):
    """
    An image transformer adapted for images and videos with both fixed and dynamic resolution.
    """

    def __init__(self, config):
        self.config = config
        self.resolution_mode = config.resolution_mode

        self.image_mean, self.image_std = config.image_mean, config.image_std
        self.patch_size = config.patch_size  # spatial patch size
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.resize_factor = (
            config.patch_size * config.spatial_merge_size * config.resize_factor
        )
        self.relarge_ratio = config.relarge_ratio

        self.v_post_squeeze = config.v_post_squeeze  # video post-compression.
        self.video_vit = config.video_vit

        self.forced_transform = None
        self.min_pixels, self.max_pixels = None, None
        self.min_pixels_video, self.max_pixels_video = None, None
        assert self.resolution_mode in [
            "native",
            "224",
            "378",
            "756",
        ], "Only provide data processing pipelines for these mode."
        if self.resolution_mode == "native":
            self.min_pixels = config.min_tokens * config.patch_size * config.patch_size
            self.max_pixels = config.max_tokens * config.patch_size * config.patch_size
            self.min_pixels_video = (
                getattr(config, "min_tokens_video", config.min_tokens)
                * config.patch_size
                * config.patch_size
            )
            self.max_pixels_video = (
                getattr(config, "max_tokens_video", config.max_tokens)
                * config.patch_size
                * config.patch_size
            )
            self.min_pixels_video_vit = (
                getattr(config, "min_tokens_video_vit", config.min_tokens)
                * config.patch_size
                * config.patch_size
            )
            self.max_pixels_video_vit = (
                getattr(config, "max_tokens_video_vit", config.max_tokens)
                * config.patch_size
                * config.patch_size
            )
        else:
            image_size = int(self.resolution_mode)
            self.forced_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    self.convert_to_rgb,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.image_mean, std=self.image_std),
                ]
            )

    def __call__(self, images, is_video=False):
        if not isinstance(images, List):
            images = [images]  # shape of each image is [h, w, c]
        sample_num = len(images)
        is_single = sample_num == 1
        assert self.relarge_ratio == 1, "The relarge operation is not supported."

        if self.resolution_mode == "native":
            if is_single:
                min_pixels, max_pixels = (self.min_pixels, self.max_pixels)
            elif self.video_vit and is_video:
                assert (
                    sample_num % 2 == 0
                ), f"In video_vit mode, {sample_num=} % 2 != 0 indicates abnormal data."
                min_pixels, max_pixels = (
                    self.min_pixels_video_vit,
                    self.max_pixels_video_vit // (sample_num // 2),
                )
                max_pixels = min(max_pixels, self.max_pixels)
            else:
                min_pixels, max_pixels = (
                    self.min_pixels_video_vit,
                    self.max_pixels_video_vit // sample_num,
                )
                max_pixels = min(max_pixels, self.max_pixels)

            pixel_range_list = [(min_pixels, max_pixels)] * sample_num
            if not is_single and (not self.v_post_squeeze):
                pixel_nums_list = [
                    np.prod(
                        self.smart_resize(
                            image.size[1],
                            image.size[0],
                            self.resize_factor,
                            min_pixels,
                            max_pixels,
                        )
                    )
                    for image in images
                ]
                if (pixel_num_sum := sum(pixel_nums_list)) / max_pixels > 1:
                    pixel_range_list = [
                        (
                            min_pixels * cur_pixel_nums // pixel_num_sum,
                            max_pixels * cur_pixel_nums // pixel_num_sum,
                        )
                        for cur_pixel_nums in pixel_nums_list
                    ]
            # Perform image transformation on a single frame.
            processed_images = []
            for i in range(sample_num):
                image = images[i]
                min_pixels, max_pixels = pixel_range_list[i]
                max_pixels = min(max_pixels, self.max_pixels)
                resized_height, resized_width = self.smart_resize(
                    image.size[1],
                    image.size[0],
                    self.resize_factor,
                    min_pixels,
                    max_pixels,
                )
                image = self.convert_to_rgb(image)
                image = self.resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=Image.Resampling.BICUBIC,
                )
                image = self.rescale(image, scale=1 / 255)
                image = self.normalize(
                    image=image, mean=self.image_mean, std=self.image_std
                )

                image = image.transpose(2, 0, 1)  # c,h,w
                if not (self.video_vit and is_video and not is_single):
                    image = np.tile(
                        image, (self.temporal_patch_size, 1, 1, 1)
                    )  # 2,c,h,w
                processed_images.append(image)
        else:
            images = [self.forced_transform(image).numpy() for image in images]
            processed_images = [
                np.tile(image, (self.temporal_patch_size, 1, 1, 1)) for image in images
            ]

        if self.video_vit and is_video and not is_single:
            new_processed_images = []
            for i in range(len(processed_images) // 2):
                unit_processed_images = processed_images[2 * i : 2 * i + 2]
                unit_processed_images = np.stack(unit_processed_images)  # 2,c,h,w
                new_processed_images.append(unit_processed_images)
            processed_images = new_processed_images

        processed_images = [torch.from_numpy(image) for image in processed_images]
        return processed_images

    @staticmethod
    def convert_to_rgb(image):
        if not isinstance(image, PIL.Image.Image):
            return image
        # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
        # for transparent images. The call to `alpha_composite` handles this case
        if image.mode == "RGB":
            return image
        image_rgba = image.convert("RGBA")
        background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image_rgba)
        alpha_composite = alpha_composite.convert("RGB")
        return alpha_composite

    @staticmethod
    def resize(image, size, resample, return_numpy: bool = True) -> np.ndarray:
        """
        Resizes `image` to `(height, width)` specified by `size` using the PIL library.
        """
        if not len(size) == 2:
            raise ValueError("size must have 2 elements")
        assert isinstance(image, PIL.Image.Image)
        height, width = size
        resample = resample if resample is not None else PIL.Image.Resampling.BILINEAR
        # PIL images are in the format (width, height)
        resized_image = image.resize(
            (width, height), resample=resample, reducing_gap=None
        )
        if return_numpy:
            resized_image = np.array(resized_image)
            resized_image = (
                np.expand_dims(resized_image, axis=-1)
                if resized_image.ndim == 2
                else resized_image
            )
        return resized_image

    @staticmethod
    def rescale(
        image: np.ndarray, scale: float, dtype: np.dtype = np.float32
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Input image must be of type np.ndarray, got {type(image)}"
            )
        rescaled_image = image * scale
        rescaled_image = rescaled_image.astype(dtype)
        return rescaled_image

    @staticmethod
    def normalize(image, mean, std) -> np.ndarray:
        """
        Normalizes `image` using the mean and standard deviation specified by `mean` and `std`.
        image = (image - mean) / std
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        num_channels = image.shape[-1]
        # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
        # We preserve the original dtype if it is a float type to prevent upcasting float16.
        if not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
        if isinstance(mean, Iterable):
            if len(mean) != num_channels:
                raise ValueError(
                    f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}"
                )
        else:
            mean = [mean] * num_channels
        mean = np.array(mean, dtype=image.dtype)
        if isinstance(std, Iterable):
            if len(std) != num_channels:
                raise ValueError(
                    f"std must have {num_channels} elements if it is an iterable, got {len(std)}"
                )
        else:
            std = [std] * num_channels
        std = np.array(std, dtype=image.dtype)
        image = (image - mean) / std
        return image

    @staticmethod
    def smart_resize(height, width, factor, min_pixels, max_pixels):
        """
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if height < factor or width < factor:
            if height < factor:
                ratio = factor / height
                height, width = factor, int(ratio * width) + 1
            if width < factor:
                ratio = factor / width
                width, height = factor, int(ratio * height) + 1
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        if h_bar < factor:  # Extreme aspect ratio case.
            h_bar, w_bar = factor, max_pixels // factor
        if w_bar < factor:
            h_bar, w_bar = max_pixels // factor, factor
        if h_bar % 14 != 0:
            h_bar = h_bar // 14 * 14
        if w_bar % 28 != 0:
            w_bar = w_bar // 28 * 28
        return h_bar, w_bar
