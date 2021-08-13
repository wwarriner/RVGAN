import argparse
from pathlib import Path, PurePath
from typing import Dict

import numpy as np

import src.data
import src.file_util
import src.image_util
import src.model


def eval(
    image_chunks: np.ndarray,
    mask_chunks: np.ndarray,
    downscale_factor: int,
    g_c: src.data.ModelFile,
    g_f: src.data.ModelFile,
) -> np.ndarray:
    dummy_label_chunks = np.zeros_like(mask_chunks)
    dummy_images_per_batch = 1
    dataset = src.data.Dataset(
        XA_fr=image_chunks,
        XB_fr=mask_chunks,
        XC_fr=dummy_label_chunks,
        downscale_factor=downscale_factor,
        images_per_batch=dummy_images_per_batch,
        g_f_arch=g_f.model,
        g_c_arch=g_c.model,
    )
    data = dataset.get_full_data()
    out = data["XC_fx"]
    return out  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--image_extension", type=str, default=".png")
    parser.add_argument("--mask_extension", type=str, default=".png")
    args = parser.parse_args()

    model_folder = PurePath(args.model_folder)
    assert src.file_util.check_folder(model_folder)

    input_folder = PurePath(args.input_folder)
    assert src.file_util.check_folder(input_folder)

    output_folder = PurePath(args.output_folder)
    # no check, this gets created

    config_file = PurePath(args.config_file)
    assert src.file_util.check_file(config_file)

    image_extension = src.file_util.fix_ext(args.image_extension)
    mask_extension = src.file_util.fix_ext(args.mask_extension)

    config = src.file_util.read_yaml(config_file)
    input_shape_px = np.array(config["arch"]["input_size"])
    downscale_factor = config["arch"]["downscale_factor"]

    # LOAD MODELS
    print("loading models...")
    arch_factory = src.model.ArchFactory(
        input_shape_px=input_shape_px, downscale_factor=downscale_factor
    )

    g_c_arch = arch_factory.build_generator(scale_type="coarse")
    g_c = src.data.ModelFile(name="g_c", folder=model_folder, arch=g_c_arch)
    g_c.load(version="latest")

    g_f_arch = arch_factory.build_generator(scale_type="fine")
    g_f = src.data.ModelFile(name="g_f", folder=model_folder, arch=g_f_arch)
    g_f.load(version="latest")

    # LOAD AND PROCESS IMAGES
    print("evaluating...")
    image_files = src.file_util.glob(
        folder=input_folder / "image", pattern="*" + image_extension
    )
    mask_files = src.file_util.glob(
        folder=input_folder / "mask", pattern="*" + mask_extension
    )

    for image_file, mask_file in zip(image_files, mask_files):
        print(str(image_file))

        image = src.image_util.load_image(path=image_file)
        image = src.image_util.intensity_to_input(image=image)
        image_chunks = src.image_util.image_to_chunks(
            image, chunk_shape_px=input_shape_px, stride_px=input_shape_px
        )

        mask = src.image_util.load_image(path=mask_file)
        mask = src.image_util.binary_to_input(image=mask)
        mask_chunks = src.image_util.image_to_chunks(
            mask, chunk_shape_px=input_shape_px, stride_px=input_shape_px
        )

        label_chunks = eval(
            image_chunks=image_chunks,
            mask_chunks=mask_chunks,
            downscale_factor=downscale_factor,
            g_c=g_c,
            g_f=g_f,
        )

        image_shape_space_px = src.image_util.get_shape_space_px(image=image)
        label = src.image_util.chunks_to_image(
            chunks=label_chunks,
            image_shape_space_px=image_shape_space_px,
            stride_px=input_shape_px,
        )
        label = src.image_util.output_to_binary(image=label, threshold=0.5)

        Path(output_folder).mkdir(parents=True, exist_ok=True)
        label_file = output_folder / (image_file.stem + ".png")
        src.image_util.save_image(path=label_file, image=label)
