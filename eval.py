import argparse
from pathlib import Path, PurePath

import numpy as np
import yaml

import preprocess
import src.data
import src.dataloader
import src.model


def eval(
    g_global_model: src.data.ModelFile,
    g_local_model: src.data.ModelFile,
    image_chunks: np.ndarray,
    mask_chunks: np.ndarray,
) -> np.ndarray:
    coarse_shape_space = (
        image_chunks.shape[1] // 2,
        image_chunks.shape[2] // 2,
    )  # TODO fix magic numbers
    image_chunks_coarse = src.dataloader.resize_stack(
        data=image_chunks, out_shape_space=coarse_shape_space
    )
    mask_chunks_coarse = src.dataloader.resize_stack(
        data=mask_chunks, out_shape_space=coarse_shape_space
    )

    # GLOBAL IS FINE
    # LOCAL IS COARSE

    [_, x_global], _ = src.dataloader.generate_fake_data_coarse(
        g_local_model.model, image_chunks_coarse, mask_chunks_coarse, [1, 1],
    )
    out, _ = src.dataloader.generate_fake_data_fine(
        g_global_model.model, image_chunks, mask_chunks, x_global, [1, 1]
    )
    return out  # type: ignore


# TODO def reassemble full images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--image_extension", type=str, default=".png")
    parser.add_argument("--mask_extension", type=str, default=".png")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # LOAD MODELS
    print("loading models...")
    arch_factory = src.model.ArchFactory(
        input_size=config["arch"]["input_size"],
        downscale_factor=config["arch"]["downscale_factor"],
    )

    g_model_fine = arch_factory.build_generator(scale_type="fine")
    g_fine_file = src.data.ModelFile(
        name="global_model", folder=args.model_folder, arch=g_model_fine
    )
    g_fine_file.load(version="latest")

    g_model_coarse = arch_factory.build_generator(scale_type="coarse")
    g_coarse_file = src.data.ModelFile(
        name="local_model", folder=args.model_folder, arch=g_model_coarse
    )
    g_coarse_file.load(version="latest")

    # LOAD AND PROCESS IMAGES
    print("evaluating...")
    input_shape_px = np.array(config["arch"]["input_size"])
    input_folder = PurePath(args.input_folder)
    label_folder = PurePath(args.output_folder)
    Path(label_folder).mkdir(parents=True, exist_ok=True)

    image_files = Path(input_folder / "image").glob(pattern="*" + args.image_extension)
    image_files = sorted(list(image_files))

    mask_files = Path(input_folder / "mask").glob(pattern="*" + args.mask_extension)
    mask_files = sorted(list(mask_files))

    for image_file, mask_file in zip(image_files, mask_files):
        print(str(image_file))

        image = src.data.load_image(image_file)
        image = src.dataloader.intensity_to_input(image)

        mask = src.data.load_image(mask_file)
        mask = src.dataloader.binary_to_input(mask)

        image_chunks = preprocess.image_to_chunks(
            image, chunk_shape_px=input_shape_px, stride_px=input_shape_px
        )
        mask_chunks = preprocess.image_to_chunks(
            mask, chunk_shape_px=input_shape_px, stride_px=input_shape_px
        )

        label_chunks = eval(
            g_global_model=g_fine_file,
            g_local_model=g_coarse_file,
            image_chunks=image_chunks,
            mask_chunks=mask_chunks,
        )

        image_shape_space_px = preprocess.get_shape_space_px(image=image)
        label = preprocess.chunks_to_image(
            chunks=label_chunks,
            image_shape_space_px=image_shape_space_px,
            stride_px=input_shape_px,
        )
        label = src.dataloader.output_to_binary(image=label, threshold=0.5)

        label_file = label_folder / (image_file.stem + ".png")
        src.data.save_image(image=label, path=label_file)
