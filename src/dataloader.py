import keras.models
import numpy as np

import src.image_util


def load_real_data(filename):
    data = np.load(filename)
    X1 = data["arr_0"]  # type: ignore
    X2 = data["arr_1"]  # type: ignore
    X3 = data["arr_2"]  # type: ignore

    assert isinstance(X1, np.ndarray)
    assert isinstance(X2, np.ndarray)
    assert isinstance(X3, np.ndarray)

    X1 = src.image_util.intensity_to_input(X1)
    X2 = src.image_util.binary_to_input(X2)
    X3 = src.image_util.binary_to_input(X3)

    return [X1, X2, X3]


def generate_real_data(data, batch_id, batch_size, patch_shape):
    trainA, trainB, trainC = data

    start = batch_id * batch_size
    end = start + batch_size
    X1, X2, X3 = trainA[start:end], trainB[start:end], trainC[start:end]

    y1 = -np.ones((batch_size, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((batch_size, patch_shape[1], patch_shape[1], 1))
    return [X1, X2, X3], [y1, y2]


def generate_real_data_random(data, random_samples, patch_shape):
    trainA, trainB, trainC = data

    index = np.random.randint(0, trainA.shape[0], random_samples)
    X1, X2, X3 = trainA[index], trainB[index], trainC[index]

    y1 = -np.ones((random_samples, patch_shape[0], patch_shape[0], 1))
    y2 = -np.ones((random_samples, patch_shape[1], patch_shape[1], 1))
    return [X1, X2, X3], [y1, y2]


def generate_fake_data_fine(
    g_model: keras.models.Model, batch_data, batch_mask, x_global, patch_shape
):
    X = g_model.predict([batch_data, batch_mask, x_global])
    y1 = np.ones((len(X), patch_shape[0], patch_shape[0], 1))

    return X, y1


def generate_fake_data_coarse(
    g_model: keras.models.Model, batch_data, batch_mask, patch_shape
):
    X, X_global = g_model.predict([batch_data, batch_mask])
    y1 = np.ones((len(X), patch_shape[1], patch_shape[1], 1))

    return [X, X_global], y1
