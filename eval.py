import argparse
from pathlib import Path, PurePath

import numpy as np
import yaml

import src.data
import src.dataloader
import src.file_util
import src.image_util
import src.model

# TODO GLOBAL rename to FINE
# TODO LOCAL rename to COARSE


def eval(
    g_global_model: src.data.ModelFile,
    g_local_model: src.data.ModelFile,
    image_chunks: np.ndarray,
    mask_chunks: np.ndarray,
) -> np.ndarray:
    coarse_shape_space_px = src.image_util.downscale_shape_space_px(
        in_shape_space_px=image_chunks.shape[1:3], factor=2  # TODO magic number
    )
    image_chunks_coarse = src.image_util.resize_stack(
        stack=image_chunks, out_shape_space_px=coarse_shape_space_px
    )
    mask_chunks_coarse = src.image_util.resize_stack(
        stack=mask_chunks, out_shape_space_px=coarse_shape_space_px
    )

    N_PATCH = [1, 1]
    [_, x_global], _ = src.dataloader.generate_fake_data_coarse(
        g_local_model.model, image_chunks_coarse, mask_chunks_coarse, N_PATCH,
    )
    out, _ = src.dataloader.generate_fake_data_fine(
        g_global_model.model, image_chunks, mask_chunks, x_global, N_PATCH
    )
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
        input_size=input_shape_px, downscale_factor=downscale_factor
    )

    g_model_fine = arch_factory.build_generator(scale_type="fine")
    g_fine_file = src.data.ModelFile(
        name="global_model", folder=model_folder, arch=g_model_fine
    )
    g_fine_file.load(version="latest")

    g_model_coarse = arch_factory.build_generator(scale_type="coarse")
    g_coarse_file = src.data.ModelFile(
        name="local_model", folder=model_folder, arch=g_model_coarse
    )
    g_coarse_file.load(version="latest")

    # LOAD AND PROCESS IMAGES
    print("evaluating...")
    image_files = Path(input_folder / "image").glob(pattern="*" + image_extension)
    image_files = sorted(list(image_files))

    mask_files = Path(input_folder / "mask").glob(pattern="*" + mask_extension)
    mask_files = sorted(list(mask_files))

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
            g_global_model=g_fine_file,
            g_local_model=g_coarse_file,
            image_chunks=image_chunks,
            mask_chunks=mask_chunks,
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
