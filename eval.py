import argparse
from pathlib import Path, PurePath

import numpy as np
import yaml

import src.data
import src.dataloader
import src.image_util
import src.model


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

        image = src.image_util.load_image(path=image_file)
        image = src.image_util.intensity_to_input(image=image)

        mask = src.image_util.load_image(path=mask_file)
        mask = src.image_util.binary_to_input(image=mask)

        image_chunks = src.image_util.image_to_chunks(
            image, chunk_shape_px=input_shape_px, stride_px=input_shape_px
        )
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

        label_file = label_folder / (image_file.stem + ".png")
        src.image_util.save_image(path=label_file, image=label)
