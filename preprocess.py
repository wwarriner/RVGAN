import argparse
import itertools
from pathlib import Path, PurePath
from typing import List

import numpy as np

import src.image_util

# TODO add yaml config


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
    image = src.image_util.load_image(path=image_file)
    print(f"{str(image_file.name)} with shape {image.shape}")
    crops = src.image_util.image_to_chunks(
        image=image, chunk_shape_px=chunk_shape_px, stride_px=stride_px
    )
    return crops


def _save_chunk_stack(
    crops: np.ndarray, out_folder: PurePath, base_name: str, ext: str = ".png"
) -> None:
    for crop_index in range(crops.shape[0]):
        name = "_".join([base_name, str(crop_index + 1)]) + ext
        path = out_folder / name
        crop = crops[crop_index, ...]
        src.image_util.save_image(path=path, image=crop)


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
