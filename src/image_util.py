import itertools
from pathlib import Path, PurePath
from typing import List, Sequence, Union

import cv2
import numpy as np
from PIL import Image

PathLike = Union[Path, PurePath, str]


def load_image(path: PathLike) -> np.ndarray:
    image = Image.open(str(path))
    image = np.array(image)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    return image


def save_image(path: PathLike, image: np.ndarray) -> None:
    if image.shape[-1] == 1:
        image = image[..., 0]
    out = Image.fromarray(image)
    out.save(str(path))


def intensity_to_input(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image = image.astype(dtype=np.float64)
    image = (image - 127.5) / 127.5  # type: ignore
    return image


def binary_to_input(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image = image.astype(dtype=np.float64)
    if image.max() == 255.0:
        image = (image - 127.5) / 127.5  # type: ignore
    else:
        image = (image - 0.5) / 0.5  # type: ignore
    return image


def output_to_intensity(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image = (image + 1.0) / 2.0  # type: ignore
    image = 255.0 * image
    image = image.astype(dtype=np.uint8)
    return image


def output_to_binary(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    image = image.copy()
    image = (image + 1.0) / 2.0  # type: ignore
    image = image > threshold  # type: ignore
    image = image.astype(np.bool)
    return image


def downscale_shape_space_px(
    in_shape_space_px: Sequence[int], factor: int
) -> List[int]:
    out_shape_space_px = np.array(in_shape_space_px) // factor
    out_shape_space_px = list(out_shape_space_px)
    return out_shape_space_px


def resize_stack(stack: np.ndarray, out_shape_space_px: Sequence[int]) -> np.ndarray:
    """
    Uniformly resizes a stack of images to a desired shape.

    Args:
    1. stack (np.ndarray): image stack whose dimensions are index, space,
       channels
    2. out_shape_space_px (Sequence[int]): spatial shape of output, 2 elements

    Returns:
    1. (np.ndarray): resized stack
    """
    out = []
    for index in range(len(stack)):
        image = stack[index, ...].copy()
        image = resize(image=image, out_shape_space_px=out_shape_space_px)
        out.append(image)
    out = np.stack(out, axis=0)
    return out  # type: ignore


def resize(image: np.ndarray, out_shape_space_px: Sequence[int]) -> np.ndarray:
    out = cv2.resize(image, dsize=out_shape_space_px, interpolation=cv2.INTER_LANCZOS4)
    return out


def get_shape_space_px(image: np.ndarray) -> np.ndarray:
    return np.array(image.shape[0:2])


def get_shape_channels_px(image: np.ndarray) -> np.ndarray:
    return np.array(image.shape[2:])


def image_to_chunks(
    image: np.ndarray, chunk_shape_px: np.ndarray, stride_px: np.ndarray
) -> np.ndarray:
    """
    Converts a 2D image to a stack of chunks from that image. Chunks are
    extracted by striding. Image is padded on the high side with zero.

    Args:
    1. image (np.ndarray): 2D image with arbitrary channels.
    2. chunk_shape_px (np.ndarray): 2-element numpy array of integer type
       denoting size of the extracted chunks
    3. stride_px (np.ndarray): 2-element numpy array of integer type denoting
       distance to stride along spatial dimensions between chunks

    Returns:
    np.ndarray: stack of chunks
    """

    image_shape_space_px = get_shape_space_px(image=image)
    chunk_slices_px = _compute_chunk_slices_px(
        image_shape_space_px=image_shape_space_px,
        chunk_shape_px=chunk_shape_px,
        stride_px=stride_px,
    )

    padded_image = _pad_image(
        image=image.copy(), chunk_shape_px=chunk_shape_px, stride_px=stride_px
    )

    chunks = []
    for slices_px in chunk_slices_px:
        sub = (*slices_px, Ellipsis)
        chunk = padded_image[sub]
        chunks.append(chunk)
    chunks = np.stack(chunks, axis=0)
    return chunks  # type: ignore


def chunks_to_image(
    chunks: np.ndarray, image_shape_space_px: np.ndarray, stride_px: np.ndarray,
) -> np.ndarray:
    """
    Converts stack of chunks into a 2D image of input shape.

    Args:
    1. chunks (np.ndarray): stack of image chunks from 2D image
    2. image_shape_space_px (np.ndarray): 2-element numpy array of integer type
       denoting spatial shape of original 2D image
    3. stride_px (np.ndarray): 2-element numpy array of integer type denoting
       distance to stride along spatial dimensions between chunks

    Returns:
        np.ndarray: 2D image
    """

    chunk_shape_px = chunks.shape[1:3]
    shape_channel_px = chunks.shape[3:]
    image_shape_full_px = (*image_shape_space_px, *shape_channel_px)  # type: ignore
    image = np.zeros(image_shape_full_px).astype(dtype=chunks.dtype)
    padded_image = _pad_image(
        image=image, chunk_shape_px=chunk_shape_px, stride_px=stride_px
    )

    chunk_slices_px = _compute_chunk_slices_px(
        image_shape_space_px=image_shape_space_px,
        chunk_shape_px=chunk_shape_px,
        stride_px=stride_px,
    )
    for slices_px, chunk in zip(chunk_slices_px, chunks):  # type: ignore
        sub = (*slices_px, Ellipsis)
        padded_image[sub] = chunk.copy()

    out_slices = [slice(0, x) for x in image_shape_full_px]  # type: ignore
    image = padded_image[out_slices]
    return image


def _compute_chunk_counts(
    image_shape_space_px: np.ndarray, stride_px: np.ndarray
) -> np.ndarray:
    return image_shape_space_px // stride_px  # type: ignore


def _compute_chunk_slices_px(
    image_shape_space_px: np.ndarray, chunk_shape_px: np.ndarray, stride_px: np.ndarray
):
    """
    Output is a list of lists of slices. Outer list is all chunks, inner list is
    dimensions.
    """
    chunk_counts = _compute_chunk_counts(
        image_shape_space_px=image_shape_space_px, stride_px=stride_px
    )
    chunk_indices = [np.array(range(c)) for c in chunk_counts]  # type: ignore

    chunk_starts_px = [s_px * c for s_px, c in zip(stride_px, chunk_indices)]  # type: ignore
    chunk_starts_px = itertools.product(*chunk_starts_px)
    chunk_starts_px = list(chunk_starts_px)
    chunk_starts_px = [np.array(c) for c in chunk_starts_px]

    chunk_ends_px = [c + chunk_shape_px for c in chunk_starts_px]

    slices_px = []
    for start, end in zip(chunk_starts_px, chunk_ends_px):
        chunk_slices = []
        for s_x, e_x in zip(start, end):
            chunk_slices.append(slice(s_x, e_x))
        slices_px.append(chunk_slices)
    return slices_px


def _pad_image(
    image: np.ndarray, chunk_shape_px: np.ndarray, stride_px: np.ndarray
) -> np.ndarray:
    image_shape_space_px = get_shape_space_px(image=image)
    padded_shape_space_px = _compute_padded_image_shape_px(
        image_shape_space_px=image_shape_space_px,
        chunk_shape_px=chunk_shape_px,
        stride_px=stride_px,
    )
    image_shape_channels_px = get_shape_channels_px(image=image)
    padded_shape_full_px = (*padded_shape_space_px, *image_shape_channels_px)  # type: ignore
    padded_image = _pad_to_shape_space(a=image, shape_full_px=tuple(padded_shape_full_px))  # type: ignore
    return padded_image


def _compute_padded_image_shape_px(
    image_shape_space_px: np.ndarray, chunk_shape_px: np.ndarray, stride_px: np.ndarray
) -> np.ndarray:
    chunk_counts = _compute_chunk_counts(
        image_shape_space_px=image_shape_space_px, stride_px=stride_px
    )
    last_start = stride_px * chunk_counts
    last_end = last_start + chunk_shape_px
    return last_end


def _pad_to_shape_space(
    a: np.ndarray, shape_full_px: tuple, *args, **kwargs
) -> np.ndarray:
    before = np.array(a.shape)
    after = np.array(shape_full_px)
    pad = after - before
    pad = [(0, p) for p in pad]
    return np.pad(a, pad_width=pad, *args, **kwargs)
