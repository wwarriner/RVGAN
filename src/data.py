import datetime
from pathlib import Path, PurePath
from typing import Union

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.dataloader

from PIL import Image

PathLike = Union[Path, PurePath, str]


def load_image(path: PathLike) -> np.ndarray:
    image = Image.open(str(path))
    image = np.array(image)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    return image


def save_image(path: PathLike, image: np.ndarray) -> None:
    if image.shape[-1] == 1:
        image = image[..., 0]
    out = Image.fromarray(image)
    out.save(str(path))


class ModelFile:
    def __init__(self, name: str, folder: PathLike, arch: keras.models.Model):
        self._name = name
        self._folder = PurePath(folder)
        self._model = arch

    @property
    def model(self) -> keras.models.Model:
        return self._model

    def save(self, version: str) -> None:
        file_path = self._build_path(version=version)
        Path(file_path.parent).mkdir(parents=True, exist_ok=True)
        self._model.save_weights(file_path)

    def load(self, version: str) -> None:
        file_path = self._build_path(version=version)
        self._model.load_weights(file_path)

    def _build_path(self, version: str) -> PurePath:
        file_name = f"{self._name}_{version}.h5"
        file_path = PurePath(self._folder) / file_name
        return file_path


class Visualizations:
    def __init__(self, output_folder: PathLike):
        self._folder = PurePath(output_folder)

    def save_plot(
        self,
        epoch: int,
        g_global_model: ModelFile,
        g_local_model: ModelFile,
        dataset,
        n_samples=3,
    ):
        """
        A - rgb, fundus photograph
        B - binary, mask
        C - binary, label

        real - actual training data
        fake - synthetic data from GAN

        coarse/global - half scale data
        fine/local - original scale data
        """
        N_PATCH = [1, 1, 1]

        # GENERATE DATA
        # REAL
        (
            [X_realA_fine, X_realB_fine, X_realC_fine],
            _,
        ) = src.dataloader.generate_real_data_random(dataset, n_samples, N_PATCH)
        out_shape = (
            int(X_realA_fine.shape[1] / 2),  # TODO magic value
            int(X_realA_fine.shape[2] / 2),
        )  # TODO abstract this
        X_realA_coarse = src.dataloader.resize_stack(X_realA_fine, out_shape)
        X_realB_coarse = src.dataloader.resize_stack(X_realB_fine, out_shape)
        X_realC_coarse = src.dataloader.resize_stack(X_realC_fine, out_shape)
        # FAKE_COARSE
        [X_fakeC_coarse, x_global], _ = src.dataloader.generate_fake_data_coarse(
            g_global_model.model, X_realA_coarse, X_realB_coarse, N_PATCH
        )
        # FAKE_FINE
        X_fakeC_fine, _ = src.dataloader.generate_fake_data_fine(
            g_local_model.model, X_realA_fine, X_realB_fine, x_global, N_PATCH
        )

        # SAVE PLOTS
        base_name = f"{epoch:0>5d}.png"
        # FINE/LOCAL
        # scale all pixels from [-1,1] to [0,1]
        X_realA_fine = (X_realA_fine + 1) / 2.0
        X_realC_fine = (X_realC_fine + 1) / 2.0
        X_fakeC_fine = (X_fakeC_fine + 1) / 2.0  # type: ignore
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis("off")
            plt.imshow(X_realA_fine[i])
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis("off")
            twoD_img = X_fakeC_fine[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
            plt.axis("off")
            twoD_img = X_realC_fine[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        name = "_".join(["fine", base_name])
        file_path = self._folder / name
        plt.savefig(file_path)
        plt.close()
        # COARSE/GLOBAL
        # scale all pixels from [-1,1] to [0,1]
        X_realA_coarse = (X_realA_coarse + 1) / 2.0
        X_realC_coarse = (X_realC_coarse + 1) / 2.0
        X_fakeC_coarse = (X_fakeC_coarse + 1) / 2.0  # type: ignore
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis("off")
            plt.imshow(X_realA_coarse[i])
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis("off")
            twoD_img = X_fakeC_coarse[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
            plt.axis("off")
            twoD_img = X_realC_coarse[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        name = "_".join(["coarse", base_name])
        file_path = self._folder / name
        plt.savefig(file_path)
        plt.close()


class Statistics:
    EPOCH = "epoch"
    BATCH = "batch"
    ELAPSED = "elapsed"

    def __init__(self, output_folder: PathLike):
        self._data = pd.DataFrame()
        self._folder = PurePath(output_folder)
        self._previous_time = datetime.datetime.now()

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    @property
    def latest_epoch(self) -> int:
        if self.empty:
            out = 0
        else:
            out = self._data[self.EPOCH].sort_values().iloc[-1]
            out = out.item()  # type: ignore
            assert isinstance(out, int)
        return out

    @property
    def epoch_count(self) -> int:
        if self.empty:
            out = 0
        else:
            out = self._data[self.EPOCH].unique().size
        return out

    def start_timer(self) -> None:
        self._previous_time = datetime.datetime.now()

    def latest_batch_to_string(self) -> str:
        assert not self.empty
        row = self._data.iloc[-1, :]
        losses = row.drop(index=[self.EPOCH, self.BATCH, self.ELAPSED])
        losses = losses.to_dict()
        s = [f"epoch: {row[self.EPOCH]: <5d}", f"batch: {row[self.BATCH]: >5d}"]
        s.extend([f"{k:s}[{v: >2.4f}]" for k, v in losses.items()])
        s = " ".join(s)
        return s

    def latest_epoch_to_string(self) -> str:
        assert not self.empty
        data = self._data[self._data[self.EPOCH] == self.latest_epoch]
        agg_dict = {k: "mean" for k in data.columns if k != self.EPOCH}
        agg_dict[self.ELAPSED] = "sum"
        data = data.groupby(by=self.EPOCH)
        data = data.agg(agg_dict)
        data = data.drop(columns=self.BATCH)
        data = data.iloc[0, :]
        s = [
            f"EPOCH: {self.latest_epoch: <5d}",
            f"time : {data[self.ELAPSED]}",
        ]
        losses = data.drop(index=[self.ELAPSED])
        losses = losses.to_dict()
        s.extend([f"{k:s}[{v: >2.4f}]" for k, v in losses.items()])
        s = " ".join(s)
        return s

    def is_latest_of_column_smallest(self, column: str) -> bool:
        assert column in self._data.columns
        data = self._data[[self.EPOCH, column]]
        data = data.groupby(by=self.EPOCH)
        data = data[column]
        means = data.mean()
        means = means.reset_index(name=column)
        latest = means[self.EPOCH].iloc[-1]
        means = means.sort_values(by=column)
        smallest = means[self.EPOCH].iloc[0]
        return latest == smallest

    def append(self, epoch: int, batch: int, data: dict) -> None:
        now = datetime.datetime.now()
        previous = self._previous_time
        elapsed = now - previous
        self._previous_time = now
        data[self.EPOCH] = epoch
        data[self.BATCH] = batch
        data[self.ELAPSED] = elapsed
        entry = {elapsed: data}
        df = pd.DataFrame.from_dict(data=entry, orient="index")
        self._data = self._data.append(df)

    def save(self) -> None:
        self.save_csv()
        self.save_plot()

    def save_csv(self) -> None:
        file_path = self._csv_path()
        self._data.to_csv(file_path)

    def save_plot(self) -> None:
        data = self._data.drop(columns=[self.EPOCH, self.BATCH, self.ELAPSED])
        for column in data.columns:
            series = data[column].tolist()
            plt.plot(series, label=column)
        plt.legend()
        file_path = self._plot_path()
        plt.savefig(file_path)
        plt.close()

    def _csv_path(self) -> PurePath:
        return self._folder / "loss_data.csv"

    def _plot_path(self) -> PurePath:
        return self._folder / "loss_plot.png"

    def load(self) -> None:
        file_path = self._csv_path()
        data = pd.read_csv(file_path)
        self._data = data
