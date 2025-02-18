import argparse
from pathlib import Path, PurePath
from typing import Dict

import numpy as np

import src.data
import src.file_util
import src.image_util
import src.model


def _batch_update(
    dataset: src.data.Dataset,
    batch_index: int,
    d_f: src.data.ModelFile,
    d_c: src.data.ModelFile,
    g_c: src.data.ModelFile,
    g_f: src.data.ModelFile,
    gan: src.data.ModelFile,
) -> Dict[str, float]:
    batch_losses = {}

    # See "data.py" docs and readme for Hungarian notation meaning

    # ASSEMBLE DATA
    d_f.model.trainable = False
    d_c.model.trainable = False
    gan.model.trainable = False
    g_c.model.trainable = False
    g_f.model.trainable = False
    data = dataset.get_batch_data(batch_index=batch_index)
    [XA_fr, XB_fr, XC_fr] = data["X_fr"]
    [y1_fr, y2_fr] = data["y_fr"]
    [XA_cr, XB_cr, XC_cr] = data["X_cr"]
    XC_cx = data["XC_cx"]
    y1_cx = data["y_cx"]
    XC_fx = data["XC_fx"]
    y1_fx = data["y_fx"]
    weights_c_to_f = data["c_to_f"]

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
    dataset: src.data.Dataset,
    d_f: src.data.ModelFile,
    d_c: src.data.ModelFile,
    g_c: src.data.ModelFile,
    g_f: src.data.ModelFile,
    gan: src.data.ModelFile,
    statistics: src.data.Statistics,
    visualizations: src.data.Visualizations,
    epoch_count: int,
):
    start_epoch = statistics.latest_epoch
    if 0 < start_epoch:
        start_epoch += 1
    statistics.start_timer()

    print(f"starting at epoch {start_epoch} of {epoch_count}")
    print(f"epochs have {dataset.batch_count} batches of {dataset.images_per_batch}")

    for epoch in range(start_epoch, epoch_count):
        # BATCH LOOP
        for batch in range(dataset.batch_count):
            batch_losses = _batch_update(
                dataset=dataset,
                batch_index=batch,
                d_f=d_f,
                d_c=d_c,
                g_c=g_c,
                g_f=g_f,
                gan=gan,
            )
            statistics.append(epoch=epoch, batch=batch, data=batch_losses)
            print(statistics.latest_batch_to_string())
        print(statistics.latest_epoch_to_string())

        # SAVE
        print("saving epoch")
        statistics.save()
        visualizations.save_plot(epoch=epoch)

        VERSION = "latest"
        d_f.save(version=VERSION)
        d_c.save(version=VERSION)
        g_c.save(version=VERSION)
        g_f.save(version=VERSION)
        gan.save(version=VERSION)

        VERSION = f"eval_{epoch}"
        g_c.save(version=VERSION)
        g_f.save(version=VERSION)

    print(f"training complete")


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

    output_folder = PurePath(args.save_folder)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    config_file = PurePath(args.config_file)
    assert src.file_util.check_file(config_file)

    resume_training = args.resume_training

    print("loading config")
    config = src.file_util.read_yaml(path=config_file)
    input_shape_px = np.array(config["arch"]["input_size"])
    downscale_factor = config["arch"]["downscale_factor"]
    inner_weight = config["arch"]["inner_weight"]
    epoch_count = config["train"]["epochs"]
    images_per_batch = config["train"]["batch_size"]

    print("building model")
    arch_factory = src.model.ArchFactory(
        input_shape_px=input_shape_px, downscale_factor=downscale_factor,
    )

    print("  d_f")
    d_f_arch = arch_factory.build_discriminator(scale_type="fine", name="D1")
    d_f = src.data.ModelFile(name="d_f", folder=output_folder, arch=d_f_arch)

    print("  d_c")
    d_c_arch = arch_factory.build_discriminator(scale_type="coarse", name="D2")
    d_c = src.data.ModelFile(name="d_c", folder=output_folder, arch=d_c_arch)

    print("  g_f")
    g_f_arch = arch_factory.build_generator(scale_type="fine")
    g_f = src.data.ModelFile(name="g_f", folder=output_folder, arch=g_f_arch)

    print("  g_c")
    g_c_arch = arch_factory.build_generator(scale_type="coarse")
    g_c = src.data.ModelFile(name="g_c", folder=output_folder, arch=g_c_arch)

    print("  gan")
    gan_arch = arch_factory.build_gan(
        d_coarse=d_c_arch,
        d_fine=d_f_arch,
        g_coarse=g_c_arch,
        g_fine=g_f_arch,
        inner_weight=inner_weight,
    )
    gan = src.data.ModelFile(name="gan", folder=output_folder, arch=gan_arch)

    print("loading dataset")
    [XA_fr, XB_fr, XC_fr] = src.data.load_npz_data(path=input_npz_file)
    dataset = src.data.Dataset(
        XA_fr=XA_fr,
        XB_fr=XB_fr,
        XC_fr=XC_fr,
        downscale_factor=downscale_factor,
        images_per_batch=images_per_batch,
        g_f_arch=g_f.model,
        g_c_arch=g_c.model,
    )

    print("initializing statistics")
    statistics = src.data.Statistics(output_folder=output_folder)

    print("initializing visualizations")
    visualizations = src.data.Visualizations(
        output_folder=output_folder,
        dataset=dataset,
        downscale_factor=downscale_factor,
        sample_count=5,
        g_c=g_c,
        g_f=g_f,
    )

    if args.resume_training:
        print("resuming training")
        VERSION = "latest"
        d_f.load(version=VERSION)
        d_c.load(version=VERSION)
        g_c.load(version=VERSION)
        g_f.load(version=VERSION)
        gan.load(version=VERSION)
        statistics.load()
    else:
        print("starting training")

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
    )
    print("Training complete")
