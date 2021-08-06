import keras.models
import numpy as np
from typing import Callable, List
from pathlib import PurePath

import src.image_util


# A is RGB image data
# B is binary mask data
# C is binary label data

# d_ = discriminator
# g_ = generator
# _c = coarse
# _f = fine

# _fr = fine real
# _cr = coarse real
# _fx = fine fake
# _cx = coarse fake


def load_real_data(path: PurePath):
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


def generate_fr(dataset, batch_index, images_per_batch, patch_counts):
    trainA, trainB, trainC = dataset

    start = batch_index * images_per_batch
    end = start + images_per_batch
    XA_fr = trainA[start:end]
    XB_fr = trainB[start:end]
    XC_fr = trainC[start:end]

    y1_fr = -np.ones((images_per_batch, patch_counts[0], patch_counts[0], 1))
    y2_fr = -np.ones((images_per_batch, patch_counts[1], patch_counts[1], 1))
    return [XA_fr, XB_fr, XC_fr], [y1_fr, y2_fr]


def generate_fr_random(dataset, sample_count, patch_counts):
    trainA, trainB, trainC = dataset

    index = np.random.randint(0, trainA.shape[0], sample_count)
    XA_fr = trainA[index]
    XB_fr = trainB[index]
    XC_fr = trainC[index]

    y1_fr = -np.ones((sample_count, patch_counts[0], patch_counts[0], 1))
    y2_fr = -np.ones((sample_count, patch_counts[1], patch_counts[1], 1))
    return [XA_fr, XB_fr, XC_fr], [y1_fr, y2_fr]


def generate_fx(
    g_f_arch: keras.models.Model, XA_fr, XB_fr, weights_c_to_f, patch_counts
):
    X_fx = g_f_arch.predict([XA_fr, XB_fr, weights_c_to_f])
    y1_fx = np.ones((len(X_fx), patch_counts[0], patch_counts[0], 1))

    return X_fx, y1_fx


def generate_cx(g_c_arch: keras.models.Model, XA_cr, XB_cr, patch_counts):
    X_cx, weights_c_to_f = g_c_arch.predict([XA_cr, XB_cr])
    y1_cx = np.ones((len(X_cx), patch_counts[1], patch_counts[1], 1))

    return [X_cx, weights_c_to_f], y1_cx


def generate_cr(XA_fr, XB_fr, XC_fr, downscale_factor: int):
    out_shape_space_px = src.image_util.downscale_shape_space_px(
        in_shape_space_px=XA_fr.shape[1:3], factor=downscale_factor
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


def cycle_data(
    real_data_generator: Callable,
    downscale_factor: int,
    patch_counts: List[int],
    g_c_arch: keras.models.Model,
    g_f_arch: keras.models.Model,
):
    [XA_fr, XB_fr, XC_fr], [y1_fr, y2_fr] = real_data_generator()
    [XA_cr, XB_cr, XC_cr] = generate_cr(
        XA_fr=XA_fr, XB_fr=XB_fr, XC_fr=XC_fr, downscale_factor=downscale_factor
    )
    [XC_cx, weights_c_to_f], y1_cx = generate_cx(
        g_c_arch=g_c_arch, XA_cr=XA_cr, XB_cr=XB_cr, patch_counts=patch_counts
    )
    XC_fx, y1_fx = generate_fx(
        g_f_arch=g_f_arch,
        XA_fr=XA_fr,
        XB_fr=XB_fr,
        weights_c_to_f=weights_c_to_f,
        patch_counts=patch_counts,
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
