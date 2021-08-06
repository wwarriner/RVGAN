import argparse
import itertools
from pathlib import Path, PurePath
from typing import List

import numpy as np

import src.data


# TODO add yaml config


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

    out_slices = [slice(stop=x) for x in image_shape]  # type: ignore
    image = padded_image[out_slices]
    return image


def get_shape_space_px(image: np.ndarray) -> np.ndarray:
    return np.array(image.shape[0:2])


def get_shape_channels_px(image: np.ndarray) -> np.ndarray:
    return np.array(image.shape[2:])


def _chunk_image_files(
    files: list, chunk_shape_px: np.ndarray, stride_px: np.ndarray
) -> List[np.ndarray]:
    all_crops = []
    for image_file in files:
        crops = _chunk_image_file(
            image_file=image_file, chunk_shape_px=chunk_shape_px, stride_px=stride_px
        )
        all_crops.append(crops)
    return all_crops


def _chunk_image_file(
    image_file: PurePath, chunk_shape_px: np.ndarray, stride_px: np.ndarray
) -> np.ndarray:
    image = src.data.load_image(image_file)
    print(f"{str(image_file.name)} with shape {image.shape}")
    crops = image_to_chunks(
        image=image, chunk_shape_px=chunk_shape_px, stride_px=stride_px
    )
    return crops


def _save_chunks(image_type: str, info: dict, chunks: List[np.ndarray]) -> None:
    out_root = info["out"]
    out_folder = out_root / image_type
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    files = info["files"]
    for image_file, crops in zip(files, chunks):
        out_base_name = "_".join([image_file.stem, image_type])
        _save_chunk_stack(
            crops=crops, out_folder=out_folder, base_name=out_base_name, ext=".png",
        )


def _save_chunk_stack(
    crops: np.ndarray, out_folder: PurePath, base_name: str, ext: str = ".png"
) -> None:
    for crop_index in range(crops.shape[0]):
        name = "_".join([base_name, str(crop_index + 1)]) + ext
        path = out_folder / name
        crop = crops[crop_index, ...]
        src.data.save_image(path=path, image=crop)


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
            chunk_slices.append(slice(start=s_x, stop=e_x))
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
    padded_image = _pad_to_shape_space(a=image, shape=tuple(padded_shape_full_px))  # type: ignore
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    input_folder = PurePath(args.input_folder)
    output_folder = PurePath(args.output_folder)
    chunk_shape_px = np.array((args.input_dim, args.input_dim))
    stride_px = np.array((args.stride, args.stride))

    GLOB = "*.png"
    IMAGE_TYPES = ["image", "mask", "label"]

    # build file info for looping
    file_info = {}
    for image_type in IMAGE_TYPES:
        files = list(Path(input_folder / image_type).glob(GLOB))
        files = sorted(files)
        file_info[image_type] = {"files": files, "out": output_folder}

    print("Chunking images")
    chunk_data = {}
    for image_type, info in file_info.items():
        files = info["files"]
        print(f"Processing {image_type} with {len(files)} images")
        chunk_data[image_type] = _chunk_image_files(
            files=files, chunk_shape_px=chunk_shape_px, stride_px=stride_px,
        )

    keys = chunk_data.keys()
    gen = itertools.product(
        itertools.islice(keys, 0, 1), itertools.islice(keys, 1, None)
    )
    for first, other in gen:
        lhs_shape = np.array(chunk_data[first][0].shape[1:3])
        rhs_shape = np.array(chunk_data[other][0].shape[1:3])
        ok = np.all(lhs_shape == rhs_shape)
        if not ok:
            print(
                f"Mismatched spatial shapes {first}-{lhs_shape} vs {other}-{rhs_shape}"
            )

    print("Writing chunks as images")
    for image_type, chunks in chunk_data.items():
        info = file_info[image_type]
        _save_chunks(image_type=image_type, info=info, chunks=chunks)

    print("Writing npz file")
    npz_file_path = output_folder / "image_data.npz"
    stacked = {k: np.concatenate(v, axis=0) for k, v in chunk_data.items()}
    out = [stacked[x] for x in IMAGE_TYPES]  # order matters
    for i in range(len(out)):
        stack = out[i]
        if stack.ndim == 3:  # type: ignore
            out[i] = stack[..., np.newaxis]  # type: ignore
    [print(out[x].shape) for x in range(len(out))]  # type: ignore
    np.savez_compressed(npz_file_path, *out)

    print("Done")
