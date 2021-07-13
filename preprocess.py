import argparse
import itertools
from pathlib import Path, PurePath
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image


def _pad_to_shape(a: np.ndarray, shape: tuple, *args, **kwargs) -> np.ndarray:
    before = np.array(a.shape)
    after = np.array(shape)
    pad = after - before
    pad = [(0, p) for p in pad]
    return np.pad(a, pad_width=pad, *args, **kwargs)


def stride_crop(
    image: np.ndarray, crop_size_px: Tuple[int, int], stride_px: int = 1
) -> np.ndarray:
    """
    Inputs:
    1. image - [M,N,...] ndarray representing a single image
    2. crop_size - 2-ple of ints representing the size of output cropped images
    3. stride - positive integer representing how many px to stride between
        crops

    Output:
    crop - [K,M,N,...] ndarray representing a stack of K crops from image
    """
    crop_size_px = np.array(crop_size_px)
    image_size_px = np.array(image.shape[0:2])
    crop_counts = np.ceil(image_size_px / crop_size_px).astype(np.int64)  # type: ignore
    target_px = crop_counts * crop_size_px
    target_shape = [*(target_px.tolist()), *(image.shape[2:])]
    image = _pad_to_shape(
        a=image,
        shape=target_shape,  # type: ignore
        mode="constant",
        constant_values=0,
    )
    crop_indices = [range(c) for c in crop_counts.tolist()]
    crop_list = []
    for crop_coords in itertools.product(*crop_indices):
        crop_coords = np.array(crop_coords)
        start = crop_coords * stride_px
        end = start + crop_size_px
        crop_list.append(image[start[0] : end[0], start[1] : end[1], ...])
        # print(image[start[0] : end[0], start[1] : end[1], ...].shape)
    crop = np.stack(crop_list, axis=0)
    return crop  # type: ignore


def load_crops(
    image_file: PurePath, crop_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    image = np.array(Image.open(str(image_file)))
    print(f"{str(image_file.name)} with shape {image.shape}")
    crops = crop_fn(image)
    return crops


def crop_images(image_type: str, info: dict, crop_fn: Callable) -> List[np.ndarray]:
    files = info["files"]
    print(f"Processing {image_type} with {len(files)} images")
    all_crops = []
    for image_file in files:
        crops = load_crops(image_file=image_file, crop_fn=crop_fn)
        all_crops.append(crops)
    return all_crops


def save_cropped_images(image_type: str, info: dict, cropped: List[np.ndarray]) -> None:
    out_root = info["out"]
    out_folder = out_root / image_type
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    files = info["files"]
    for image_file, crops in zip(files, cropped):
        out_base_name = "_".join([image_file.stem, image_type])
        save_crop_stack(
            crops=crops, out_folder=out_folder, base_name=out_base_name, ext=".png",
        )


def save_crop_stack(
    crops: np.ndarray, out_folder: PurePath, base_name: str, ext: str = ".png"
) -> None:
    for crop_index in range(crops.shape[0]):
        name = "_".join([base_name, str(crop_index + 1)]) + ext
        path = out_folder / name
        crop = crops[crop_index, ...]
        out = Image.fromarray(crop)
        out.save(fp=str(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    args = parser.parse_args()

    input_folder = PurePath(args.input_folder)
    output_folder = PurePath(args.output_folder)
    crop_size_px = (args.input_dim, args.input_dim)
    stride_px = args.stride

    GLOB = "*.png"
    IMAGE_TYPES = ["image", "mask", "label"]

    # build file info for looping
    file_info = {}
    for image_type in IMAGE_TYPES:
        files = list(Path(input_folder / image_type).glob(GLOB))
        files = sorted(files)
        file_info[image_type] = {"files": files, "out": output_folder}

    print("Cropping images")
    crop_fn = lambda x: stride_crop(
        image=x, crop_size_px=crop_size_px, stride_px=stride_px
    )
    crop_data = {}
    for image_type, info in file_info.items():
        crop_data[image_type] = crop_images(
            image_type=image_type, info=info, crop_fn=crop_fn
        )

    print("Checking cropped data")
    keys = crop_data.keys()
    gen = itertools.product(
        itertools.islice(keys, 0, 1), itertools.islice(keys, 1, None)
    )
    for first, other in gen:
        lhs_shape = np.array(crop_data[first][0].shape[1:3])
        rhs_shape = np.array(crop_data[other][0].shape[1:3])
        ok = np.all(lhs_shape == rhs_shape)
        if not ok:
            print(
                f"Mismatched spatial shapes {first}-{lhs_shape} vs {other}-{rhs_shape}"
            )

    print("Writing cropped images to disk")
    for image_type, cropped in crop_data.items():
        info = file_info[image_type]
        save_cropped_images(image_type=image_type, info=info, cropped=cropped)

    print("Writing npz to disk")
    npz_file_path = output_folder / "image_data.npz"
    stacked = {k: np.concatenate(v, axis=0) for k, v in crop_data.items()}
    out = [stacked[x] for x in IMAGE_TYPES]  # order matters
    for i in range(len(out)):
        stack = out[i]
        if stack.ndim == 3:
            out[i] = stack[..., np.newaxis]
    [print(out[x].shape) for x in range(len(out))]
    np.savez_compressed(npz_file_path, *out)

    print("Done")
