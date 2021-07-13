import argparse
import gc
from pathlib import Path
from typing import Dict, List

import keras.backend as K

import src.data
from src.dataloader import (
    generate_fake_data_coarse,
    generate_fake_data_fine,
    generate_real_data,
    load_real_data,
    resize,
)
from src.model import RVgan, coarse_generator, discriminator_ae, fine_generator


def batch_update(
    batch: int,
    n_batch: int,
    n_patch: List[int],
    d_model1: src.data.ModelFile,
    d_model2: src.data.ModelFile,
    g_global_model: src.data.ModelFile,
    g_local_model: src.data.ModelFile,
    gan_model: src.data.ModelFile,
) -> Dict[str, float]:
    batch_losses = {}

    # UPDATE DISCRIMINATORS
    d_model1.model.trainable = True
    d_model2.model.trainable = True
    gan_model.model.trainable = False
    g_global_model.model.trainable = False
    g_local_model.model.trainable = False
    for j in range(2):
        # select a batch of real samples
        [X_realA, X_realB, X_realC], [y1, y2] = generate_real_data(
            dataset, batch, n_batch, n_patch
        )

        # generate a batch of fake samples for Coarse Generator
        out_shape = (int(X_realA.shape[1] / 2), int(X_realA.shape[2] / 2))
        [X_realA_half, X_realB_half, X_realC_half] = resize(
            X_realA, X_realB, X_realC, out_shape
        )
        [X_fakeC_half, x_global], y1_coarse = generate_fake_data_coarse(
            g_global_model.model, X_realA_half, X_realB_half, n_patch
        )

        # generate a batch of fake samples for Fine Generator
        X_fakeC, y1_fine = generate_fake_data_fine(
            g_local_model.model, X_realA, X_realB, x_global, n_patch
        )

        ## FINE DISCRIMINATOR
        # update discriminator for real samples
        d_loss1 = d_model1.model.train_on_batch([X_realA, X_realC], y1)[0]
        # update discriminator for generated samples
        d_loss2 = d_model1.model.train_on_batch([X_realA, X_fakeC], y1_fine)[0]

        # d_loss1 = 0.5*(d_loss1_real[0]+d_loss1_fake[0])

        # d_loss2 = 0.5*(d_loss2_real[0]+d_loss2_fake[0])

        ## COARSE DISCRIMINATOR
        # update discriminator for real samples
        d_loss3 = d_model2.model.train_on_batch([X_realA_half, X_realC_half], y2)[0]
        # update discriminator for generated samples
        d_loss4 = d_model2.model.train_on_batch(
            [X_realA_half, X_fakeC_half], y1_coarse
        )[0]
    batch_losses.update({"d1": d_loss1, "d2": d_loss2, "d3": d_loss3, "d4": d_loss4})

    # UPDATE GLOBAL GENERATOR
    d_model1.model.trainable = False
    d_model2.model.trainable = False
    gan_model.model.trainable = False
    g_global_model.model.trainable = True
    g_local_model.model.trainable = False

    # select a batch of real samples for Local enhancer
    [X_realA, X_realB, X_realC], _ = generate_real_data(
        dataset, batch, n_batch, n_patch
    )

    # Global Generator image fake and real
    out_shape = (int(X_realA.shape[1] / 2), int(X_realA.shape[2] / 2))
    [X_realA_half, X_realB_half, X_realC_half] = resize(
        X_realA, X_realB, X_realC, out_shape
    )
    [X_fakeC_half, x_global], _ = generate_fake_data_coarse(
        g_global_model.model, X_realA_half, X_realB_half, n_patch
    )

    # update the global generator
    batch_losses["g_global"], _ = g_global_model.model.train_on_batch(
        [X_realA_half, X_realB_half], [X_realC_half]
    )

    # UPDATE LOCAL ENHANCER
    d_model1.model.trainable = False
    d_model2.model.trainable = False
    gan_model.model.trainable = False
    g_global_model.model.trainable = False
    g_local_model.model.trainable = True

    # update the Local Enhancer
    batch_losses["g_local"] = g_local_model.model.train_on_batch(
        [X_realA, X_realB, x_global], X_realC
    )

    # UPDATE GAN
    d_model1.model.trainable = False
    d_model2.model.trainable = False
    gan_model.model.trainable = True
    g_global_model.model.trainable = True
    g_local_model.model.trainable = True
    # update the generator
    (
        gan_loss,
        _,
        _,
        fm1_loss,
        fm2_loss,
        _,
        _,
        g_global_recon_loss,
        g_local_recon_loss,
    ) = gan_model.model.train_on_batch(
        [
            X_realA,
            X_realA_half,
            x_global,
            X_realB,
            X_realB_half,
            X_realC,
            X_realC_half,
        ],
        [y1, y2, X_fakeC, X_fakeC_half, X_fakeC_half, X_fakeC, X_fakeC_half, X_fakeC,],
    )
    batch_losses.update(
        {
            "gan": gan_loss,
            "fm1": fm1_loss,
            "fm2": fm2_loss,
            "g_global_recon": g_global_recon_loss,
            "g_local_recon": g_local_recon_loss,
        }
    )

    return batch_losses


def train(
    d_model1: src.data.ModelFile,
    d_model2: src.data.ModelFile,
    g_global_model: src.data.ModelFile,
    g_local_model: src.data.ModelFile,
    gan_model: src.data.ModelFile,
    statistics: src.data.Statistics,
    vis: src.data.Visualizations,
    dataset,
    n_epochs=20,
    n_batch=1,
    n_patch=[64, 32],
):
    trainA, _, _ = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    start_epoch = statistics.latest_epoch
    statistics.start_timer()

    for epoch in range(start_epoch, n_epochs):
        for batch in range(bat_per_epo):
            batch_losses = batch_update(
                batch,
                n_batch=n_batch,
                n_patch=n_patch,
                d_model1=d_model1,
                d_model2=d_model2,
                g_global_model=g_global_model,
                g_local_model=g_local_model,
                gan_model=gan_model,
            )
            statistics.append(epoch=epoch, batch=batch, data=batch_losses)
            print(statistics.latest_batch_to_string())
        print(statistics.latest_epoch_to_string())
        statistics.save()
        vis.save_plot(
            epoch=epoch,
            g_global_model=g_global_model,
            g_local_model=g_local_model,
            dataset=dataset,
        )
        d_model1.save(version="latest")
        d_model2.save(version="latest")
        g_global_model.save(version="latest")
        g_local_model.save(version="latest")
        gan_model.save(version="latest")
        if statistics.is_latest_of_column_smallest(column="g_global"):
            d_model1.save(version="best")
            d_model2.save(version="best")
            g_global_model.save(version="best")
            g_local_model.save(version="best")
            gan_model.save(version="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--npz_file", type=str, default="DRIVE", help="path/to/npz/file",
    )
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument(
        "--savedir",
        type=str,
        required=False,
        help="path/to/save_directory",
        default="RVGAN",
    )
    parser.add_argument(
        "--resume_training",
        type=str,
        required=False,
        default="no",
        choices=["yes", "no"],
    )
    parser.add_argument("--inner_weight", type=float, default=0.5)
    args = parser.parse_args()

    Path(args.savedir).mkdir(parents=True, exist_ok=True)

    K.clear_session()
    gc.collect()
    dataset = load_real_data(args.npz_file)
    print("Loaded", dataset[0].shape, dataset[1].shape)

    # define input shape based on the loaded dataset
    in_size = args.input_dim
    image_shape_coarse = (in_size // 2, in_size // 2, 3)
    mask_shape_coarse = (in_size // 2, in_size // 2, 1)
    label_shape_coarse = (in_size // 2, in_size // 2, 1)

    image_shape_fine = (in_size, in_size, 3)
    mask_shape_fine = (in_size, in_size, 1)
    label_shape_fine = (in_size, in_size, 1)

    image_shape_xglobal = (in_size // 2, in_size // 2, 128)
    ndf = 64
    ncf = 128
    nff = 128

    d_model1 = discriminator_ae(image_shape_fine, label_shape_fine, ndf, name="D1")
    d1_file = src.data.ModelFile(
        name="discriminator_1", folder=args.savedir, arch=d_model1
    )

    d_model2 = discriminator_ae(image_shape_coarse, label_shape_coarse, ndf, name="D2")
    d2_file = src.data.ModelFile(
        name="discriminator_2", folder=args.savedir, arch=d_model2
    )

    g_model_fine = fine_generator(
        x_coarse_shape=image_shape_xglobal,
        input_shape=image_shape_fine,
        mask_shape=mask_shape_fine,
        nff=nff,
        n_blocks=3,
    )
    g_fine_file = src.data.ModelFile(
        name="global_model", folder=args.savedir, arch=g_model_fine
    )

    g_model_coarse = coarse_generator(
        img_shape=image_shape_coarse,
        mask_shape=mask_shape_coarse,
        n_downsampling=2,
        n_blocks=9,
        ncf=ncf,
        n_channels=1,
    )
    g_coarse_file = src.data.ModelFile(
        name="local_model", folder=args.savedir, arch=g_model_coarse
    )

    rvgan_model = RVgan(
        g_model_fine,
        g_model_coarse,
        d_model1,
        d_model2,
        image_shape_fine,
        image_shape_coarse,
        image_shape_xglobal,
        mask_shape_fine,
        mask_shape_coarse,
        label_shape_fine,
        label_shape_coarse,
        args.inner_weight,
    )
    rvgan_file = src.data.ModelFile(
        name="rvgan_model", folder=args.savedir, arch=rvgan_model
    )

    stats = src.data.Statistics(output_folder=args.savedir)
    vis = src.data.Visualizations(output_folder=args.savedir)

    if args.resume_training == "yes":
        d1_file.load(version="latest")
        d2_file.load(version="latest")
        g_coarse_file.load(version="latest")
        g_fine_file.load(version="latest")
        rvgan_file.load(version="latest")
        stats.load()

    train(
        d1_file,
        d2_file,
        g_coarse_file,
        g_fine_file,
        rvgan_file,
        stats,
        vis,
        dataset,
        n_epochs=args.epochs,
        n_batch=args.batch_size,
        n_patch=[128, 64],
    )
    print("Training complete")
