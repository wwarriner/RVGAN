import datetime
from pathlib import Path, PurePath
from typing import Union

import keras.models
import matplotlib.pyplot as plt
import pandas as pd

import src.dataloader
import src.image_util

PathLike = Union[Path, PurePath, str]


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
    def __init__(
        self,
        output_folder: PathLike,
        dataset,
        downscale_factor: int,
        sample_count: int,
        g_c: ModelFile,
        g_f: ModelFile,
    ):
        self._folder = PurePath(output_folder)
        self._dataset = dataset
        self._downscale_factor = downscale_factor
        self._sample_count = sample_count
        self._g_c = g_c
        self._g_f = g_f

    def save_plot(self, epoch: int):
        """
        A - rgb, fundus photograph
        B - binary, mask
        C - binary, label

        real - actual training data
        fake - synthetic data from GAN

        coarse/global - half scale data
        fine/local - original scale data
        """

        # EXTRACT DATA
        PATCH_COUNTS = [1, 1]
        self._g_c.model.trainable = False
        self._g_f.model.trainable = False
        real_data_generator = lambda: src.dataloader.generate_fr_random(
            dataset=self._dataset,
            sample_count=self._sample_count,
            patch_counts=PATCH_COUNTS,
        )
        cycled_data = src.dataloader.cycle_data(
            real_data_generator=real_data_generator,
            downscale_factor=self._downscale_factor,
            patch_counts=PATCH_COUNTS,
            g_c_arch=self._g_c.model,
            g_f_arch=self._g_f.model,
        )
        [XA_fr, _, XC_fr] = cycled_data["X_fr"]
        [XA_cr, _, XC_cr] = cycled_data["X_cr"]
        XC_cx = cycled_data["XC_cx"]
        XC_fx = cycled_data["XC_fx"]

        # SAVE PLOTS
        base_name = f"{epoch:0>5d}.png"

        # FINE
        XA_fr = src.image_util.output_to_intensity(XA_fr)
        XC_fr = src.image_util.output_to_intensity(XC_fr)
        XC_fx = src.image_util.output_to_intensity(XC_fx)  # type: ignore
        for i in range(self._sample_count):
            plt.subplot(3, self._sample_count, 1 + i)
            plt.axis("off")
            plt.imshow(XA_fr[i])
        for i in range(self._sample_count):
            plt.subplot(3, self._sample_count, 1 + self._sample_count + i)
            plt.axis("off")
            twoD_img = XC_fx[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        for i in range(self._sample_count):
            plt.subplot(3, self._sample_count, 1 + self._sample_count * 2 + i)
            plt.axis("off")
            twoD_img = XC_fr[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        name = "_".join(["fine", base_name])
        file_path = self._folder / name
        plt.savefig(file_path)
        plt.close()

        # COARSE
        XA_cr = src.image_util.output_to_intensity(XA_cr)  # type: ignore
        XC_cr = src.image_util.output_to_intensity(XC_cr)  # type: ignore
        XC_cx = src.image_util.output_to_intensity(XC_cx)  # type: ignore
        for i in range(self._sample_count):
            plt.subplot(3, self._sample_count, 1 + i)
            plt.axis("off")
            plt.imshow(XA_cr[i])
        for i in range(self._sample_count):
            plt.subplot(3, self._sample_count, 1 + self._sample_count + i)
            plt.axis("off")
            twoD_img = XC_cx[:, :, :, 0]
            plt.imshow(twoD_img[i], cmap="gray")
        for i in range(self._sample_count):
            plt.subplot(3, self._sample_count, 1 + self._sample_count * 2 + i)
            plt.axis("off")
            twoD_img = XC_cr[:, :, :, 0]
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
