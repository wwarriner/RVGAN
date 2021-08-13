import datetime
from pathlib import Path, PurePath
from typing import Callable, List, Union

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.image_util

PathLike = Union[Path, PurePath, str]


class Dataset:
    """
    A is RGB image data
    B is binary mask data
    C is binary label data

    d_ = discriminator
    g_ = generator
    _c = coarse
    _f = fine

    _fr = fine real
    _cr = coarse real
    _fx = fine fake
    _cx = coarse fake
    """

    def __init__(
        self,
        XA_fr,
        XB_fr,
        XC_fr,
        downscale_factor: int,
        images_per_batch: int,
        g_f_arch: keras.models.Model,
        g_c_arch: keras.models.Model,
    ):
        image_shape_px_f = np.array(XA_fr.shape[1:3])
        image_shape_px_c = src.image_util.downscale_shape_space_px(
            in_shape_space_px=image_shape_px_f, factor=downscale_factor  # type: ignore
        )
        image_shape_px_c = np.array(image_shape_px_c)

        self._XA_fr = XA_fr
        self._XB_fr = XB_fr
        self._XC_fr = XC_fr
        self._downscale_factor = downscale_factor
        self._images_per_batch = images_per_batch
        self._image_shape_px_f = image_shape_px_f
        self._image_shape_px_c = image_shape_px_c
        self._g_f_arch = g_f_arch
        self._g_c_arch = g_c_arch

    @property
    def image_count(self) -> int:
        return self._XA_fr.shape[0]

    def shuffle(self) -> None:
        indices = self._shuffled_indices
        self._XA_fr = self._XA_fr[indices, ...]
        self._XB_fr = self._XB_fr[indices, ...]
        self._XC_fr = self._XC_fr[indices, ...]

    def get_full_data(self) -> dict:
        indices = list(range(self.image_count))
        generate_fr_func = lambda: self._generate_fr(indices=indices)
        out = self._cycle(generate_fr_func=generate_fr_func)
        return out

    def get_batch_data(self, batch_index: int) -> dict:
        start = batch_index * self._images_per_batch
        end = start + self._images_per_batch
        indices = list(range(start=start, stop=end))
        generate_fr_func = lambda: self._generate_fr(indices=indices)
        out = self._cycle(generate_fr_func=generate_fr_func)
        return out

    def get_random_sample_data(self, sample_count: int) -> dict:
        indices = np.random.randint(0, self.image_count, sample_count).tolist()
        generate_fr_func = lambda: self._generate_fr(indices=indices)
        out = self._cycle(generate_fr_func=generate_fr_func)
        return out

    def _cycle(self, generate_fr_func: Callable) -> dict:
        [XA_fr, XB_fr, XC_fr], [y1_fr, y2_fr] = generate_fr_func()
        [XA_cr, XB_cr, XC_cr] = self._generate_cr(XA_fr=XA_fr, XB_fr=XB_fr, XC_fr=XC_fr)
        [XC_cx, weights_c_to_f], y1_cx = self._generate_cx(XA_cr=XA_cr, XB_cr=XB_cr)
        XC_fx, y1_fx = self._generate_fx(
            XA_fr=XA_fr, XB_fr=XB_fr, weights_c_to_f=weights_c_to_f,
        )

        out = {
            "X_fr": [XA_fr, XB_fr, XC_fr],
            "y_fr": [y1_fr, y2_fr],
            "X_cr": [XA_cr, XB_cr, XC_cr],
            "XC_cx": XC_cx,
            "y_cx": y1_cx,
            "XC_fx": XC_fx,
            "y_fx": y1_fx,
            "c_to_f": weights_c_to_f,
        }
        return out

    def _generate_fr(self, indices: List[int]):
        batch_XA_fr = self._XA_fr[indices, ...]
        batch_XB_fr = self._XB_fr[indices, ...]
        batch_XC_fr = self._XC_fr[indices, ...]

        y1_fr = -self._init_batch_y(image_shape_px=self._image_shape_px_f)
        y2_fr = -self._init_batch_y(image_shape_px=self._image_shape_px_c)
        return [batch_XA_fr, batch_XB_fr, batch_XC_fr], [y1_fr, y2_fr]

    def _generate_cr(self, XA_fr, XB_fr, XC_fr):
        out_shape_space_px = src.image_util.downscale_shape_space_px(
            in_shape_space_px=XA_fr.shape[1:3], factor=self._downscale_factor
        )
        XA_fr = src.image_util.resize_stack(
            stack=XA_fr, out_shape_space_px=out_shape_space_px
        )
        XB_fr = src.image_util.resize_stack(
            stack=XB_fr, out_shape_space_px=out_shape_space_px
        )
        XC_fr = src.image_util.resize_stack(
            stack=XC_fr, out_shape_space_px=out_shape_space_px
        )
        return [XA_fr, XB_fr, XC_fr]

    def _generate_cx(self, XA_cr, XB_cr):
        X_cx, weights_c_to_f = self._g_c_arch.predict([XA_cr, XB_cr])
        y1_cx = self._init_batch_y(image_shape_px=self._image_shape_px_c)
        return [X_cx, weights_c_to_f], y1_cx

    def _generate_fx(self, XA_fr, XB_fr, weights_c_to_f):
        X_fx = self._g_f_arch.predict([XA_fr, XB_fr, weights_c_to_f])
        y1_fx = self._init_batch_y(image_shape_px=self._image_shape_px_f)
        return X_fx, y1_fx

    def _init_batch_y(self, image_shape_px: np.ndarray) -> np.ndarray:
        return self._init_y(
            sample_count=self._images_per_batch, image_shape_px=image_shape_px
        )

    def _init_y(self, sample_count: int, image_shape_px: np.ndarray) -> np.ndarray:
        y = np.ones((sample_count, *image_shape_px, 1))  # type: ignore
        return y

    def _shuffled_indices(self) -> np.ndarray:
        indices = np.arange(len(self._XA_fr))
        indices = np.random.shuffle(indices)
        return indices


def load_npz_data(path: PurePath):
    data = np.load(path)
    XA_fr = data["arr_0"]  # type: ignore
    XB_fr = data["arr_1"]  # type: ignore
    XC_fr = data["arr_2"]  # type: ignore

    assert isinstance(XA_fr, np.ndarray)
    assert isinstance(XB_fr, np.ndarray)
    assert isinstance(XC_fr, np.ndarray)

    XA_fr = src.image_util.intensity_to_input(XA_fr)
    XB_fr = src.image_util.binary_to_input(XB_fr)
    XC_fr = src.image_util.binary_to_input(XC_fr)

    return [XA_fr, XB_fr, XC_fr]


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
        dataset: Dataset,
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
        data = self._dataset.get_random_sample_data(sample_count=self._sample_count)
        [XA_fr, _, XC_fr] = data["X_fr"]
        [XA_cr, _, XC_cr] = data["X_cr"]
        XC_cx = data["XC_cx"]
        XC_fx = data["XC_fx"]

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
