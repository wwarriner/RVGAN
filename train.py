import argparse
import gc
from pathlib import Path, PurePath
from typing import Dict, List

import keras.backend as K
import yaml

import src.data
import src.dataloader
import src.file_util
import src.image_util
import src.model


def _batch_update(
    dataset,
    batch_index: int,
    images_per_batch: int,
    patch_counts: List[int],
    downscale_factor: int,
    d_f: src.data.ModelFile,
    d_c: src.data.ModelFile,
    g_c: src.data.ModelFile,
    g_f: src.data.ModelFile,
    gan: src.data.ModelFile,
) -> Dict[str, float]:
    batch_losses = {}

    # d_ = discriminator
    # g_ = generator
    # _c = coarse
    # _f = fine

    # _fr = fine real
    # _cr = coarse real
    # _fx = fine fake
    # _cx = coarse fake

    # ASSEMBLE DATA
    d_f.model.trainable = False
    d_c.model.trainable = False
    gan.model.trainable = False
    g_c.model.trainable = False
    g_f.model.trainable = False
    real_data_generator = lambda: src.dataloader.generate_fr(
        dataset, batch_index, images_per_batch, patch_counts
    )
    cycled_data = src.dataloader.cycle_data(
        real_data_generator=real_data_generator,
        downscale_factor=downscale_factor,
        patch_counts=patch_counts,
        g_c_arch=g_c.model,
        g_f_arch=g_f.model,
    )
    [XA_fr, XB_fr, XC_fr] = cycled_data["X_fr"]
    [y1_fr, y2_fr] = cycled_data["y_fr"]
    [XA_cr, XB_cr, XC_cr] = cycled_data["X_cr"]
    XC_cx = cycled_data["XC_cx"]
    y1_cx = cycled_data["y_cx"]
    XC_fx = cycled_data["XC_fx"]
    y1_fx = cycled_data["y_fx"]
    weights_c_to_f = cycled_data["c_to_f"]

    # UPDATE DISCRIMINATORS
    d_f.model.trainable = True
    d_c.model.trainable = True
    gan.model.trainable = False
    g_c.model.trainable = False
    g_f.model.trainable = False
    for _ in range(2):
        losses = {
            "d_fr": d_f.model.train_on_batch([XA_fr, XC_fr], y1_fr)[0],
            "d_fx": d_f.model.train_on_batch([XA_fr, XC_fx], y1_fx)[0],
            "d_cr": d_c.model.train_on_batch([XA_cr, XC_cr], y2_fr)[0],
            "d_cx": d_c.model.train_on_batch([XA_cr, XC_cx], y1_cx)[0],
        }
    batch_losses.update(losses)  # type: ignore

    # UPDATE COARSE GENERATOR: _cr
    d_f.model.trainable = False
    d_c.model.trainable = False
    gan.model.trainable = False
    g_c.model.trainable = True
    g_f.model.trainable = False
    batch_losses["g_c"], _ = g_c.model.train_on_batch([XA_cr, XB_cr], [XC_cr])

    # UPDATE FINE GENERATOR: _fr
    d_f.model.trainable = False
    d_c.model.trainable = False
    gan.model.trainable = False
    g_c.model.trainable = False
    g_f.model.trainable = True
    batch_losses["g_f"] = g_f.model.train_on_batch(
        [XA_fr, XB_fr, weights_c_to_f], XC_fr
    )

    # UPDATE GAN
    d_f.model.trainable = False
    d_c.model.trainable = False
    gan.model.trainable = True
    g_c.model.trainable = True
    g_f.model.trainable = True
    (
        loss_gan,
        _,
        _,
        loss_fm_c,
        loss_fm_f,
        _,
        _,
        loss_g_c_reconstruct,
        loss_g_f_reconstruct,
    ) = gan.model.train_on_batch(
        [XA_fr, XA_cr, weights_c_to_f, XB_fr, XB_cr, XC_fr, XC_cr],
        [y1_fr, y2_fr, XC_fx, XC_cx, XC_cx, XC_fx, XC_cx, XC_fx],  # type: ignore
    )
    batch_losses.update(
        {
            "gan": loss_gan,
            "fm1": loss_fm_c,
            "fm2": loss_fm_f,
            "g_c_recon": loss_g_c_reconstruct,
            "g_f_recon": loss_g_f_reconstruct,
        }
    )

    return batch_losses


def train(
    dataset,
    d_f: src.data.ModelFile,
    d_c: src.data.ModelFile,
    g_c: src.data.ModelFile,
    g_f: src.data.ModelFile,
    gan: src.data.ModelFile,
    statistics: src.data.Statistics,
    visualizations: src.data.Visualizations,
    epoch_count: int,
    images_per_batch: int,
    patch_counts: List[int],
):
    X_A, _, _ = dataset
    batches_per_epoch = int(len(X_A) / images_per_batch)
    start_epoch = statistics.latest_epoch
    statistics.start_timer()

    for epoch in range(start_epoch, epoch_count):
        for batch in range(batches_per_epoch):
            batch_losses = _batch_update(
                dataset=dataset,
                batch_index=batch,
                images_per_batch=images_per_batch,
                patch_counts=patch_counts,
                downscale_factor=downscale_factor,
                d_f=d_f,
                d_c=d_c,
                g_c=g_c,
                g_f=g_f,
                gan=gan,
            )
            statistics.append(epoch=epoch, batch=batch, data=batch_losses)
            print(statistics.latest_batch_to_string())
        print(statistics.latest_epoch_to_string())
        statistics.save()
        visualizations.save_plot(epoch=epoch)
        VERSION = "latest"
        d_f.save(version=VERSION)
        d_c.save(version=VERSION)
        g_c.save(version=VERSION)
        g_f.save(version=VERSION)
        gan.save(version=VERSION)


# TODO save optimizer state to disk and reload
# TODO shuffle data each epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_file", type=str, required=True, help="path/to/npz/file",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        required=True,
        help="path/to/save_directory",
        default="RVGAN",
    )
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    input_npz_file = PurePath(args.npz_file)
    assert src.file_util.check_file(input_npz_file)

    output_folder = PurePath(args.savedir)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    config_file = PurePath(args.config_file)
    assert src.file_util.check_file(config_file)

    resume_training = args.resume_training

    config = src.file_util.read_yaml(path=config_file)
    input_shape_px = config["arch"]["input_size"]
    downscale_factor = config["arch"]["downscale_factor"]
    inner_weight = config["arch"]["inner_weight"]
    epoch_count = config["train"]["epochs"]
    images_per_batch = config["train"]["batch_size"]
    patch_counts = config["train"]["patch_counts"]

    dataset = src.dataloader.load_real_data(path=input_npz_file)
    print("Loaded", dataset[0].shape, dataset[1].shape)

    arch_factory = src.model.ArchFactory(
        input_shape_px=input_shape_px, downscale_factor=downscale_factor,
    )

    d_f_arch = arch_factory.build_discriminator(scale_type="fine", name="D1")
    d_f = src.data.ModelFile(
        name="discriminator_1", folder=output_folder, arch=d_f_arch
    )

    d_c_arch = arch_factory.build_discriminator(scale_type="coarse", name="D2")
    d_c = src.data.ModelFile(
        name="discriminator_2", folder=output_folder, arch=d_c_arch
    )

    g_f_arch = arch_factory.build_generator(scale_type="fine")
    g_f = src.data.ModelFile(name="fine_model", folder=output_folder, arch=g_f_arch)

    g_c_arch = arch_factory.build_generator(scale_type="coarse")
    g_c = src.data.ModelFile(name="coarse_model", folder=output_folder, arch=g_c_arch)

    rvgan_model = arch_factory.build_gan(
        d_coarse=d_c_arch,
        d_fine=d_f_arch,
        g_coarse=g_c_arch,
        g_fine=g_f_arch,
        inner_weight=inner_weight,
    )
    gan = src.data.ModelFile(name="rvgan_model", folder=output_folder, arch=rvgan_model)

    statistics = src.data.Statistics(output_folder=output_folder)
    visualizations = src.data.Visualizations(
        output_folder=output_folder,
        dataset=dataset,
        downscale_factor=downscale_factor,
        sample_count=3,
        g_c=g_c,
        g_f=g_f,
    )

    if args.resume_training:
        VERSION = "latest"
        d_f.load(version=VERSION)
        d_c.load(version=VERSION)
        g_c.load(version=VERSION)
        g_f.load(version=VERSION)
        gan.load(version=VERSION)
        statistics.load()

    train(
        dataset=dataset,
        d_f=d_f,
        d_c=d_c,
        g_c=g_c,
        g_f=g_f,
        gan=gan,
        statistics=statistics,
        visualizations=visualizations,
        epoch_count=epoch_count,
        images_per_batch=images_per_batch,
        patch_counts=patch_counts,
    )
    print("Training complete")
