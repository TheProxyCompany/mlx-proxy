# Copyright © 2023-2024 Apple Inc.

import inspect
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map
from transformers.image_processing_utils import BaseImageProcessor as ImageProcessor
from transformers.image_processing_utils import get_size_dict
from transformers.image_utils import ChannelDimension, PILImageResampling

from mlx_proxy.cache.cache import QuantizedKVCache


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

class BaseImageProcessor(ImageProcessor):
    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: dict[str, int] | None = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        if not crop_size:
            crop_size = {"height": 448, "width": 448}
        crop_size = get_size_dict(
            crop_size,
            default_to_square=True,
            param_name="crop_size",
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    @abstractmethod
    def preprocess(self, images):
        pass


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    lengths: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds < rinds
    if window_size is not None:
        mask = mask | (linds > rinds + window_size)
    if lengths is not None:
        lengths = lengths[:, None, None, None]
        mask = mask | (rinds >= lengths)
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Optional[Any] = None):
    T = h.shape[1]
    if T > 1:
        window_size = None
        offset = 0
        if cache is not None and cache[0] is not None:
            c = cache[0]
            if hasattr(c, "max_size"):
                offset = min(c.max_size, c.offset)
                window_size = c.max_size
            else:
                offset = c.offset
        mask = create_causal_mask(T, offset, window_size=window_size)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    if isinstance(cache, QuantizedKVCache):
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
