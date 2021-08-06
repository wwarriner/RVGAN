import keras.models
import numpy as np

import cv2


def intensity_to_input(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image = image.astype(dtype=np.float64)
    image = (image - 127.5) / 127.5  # type: ignore
    return image


def binary_to_input(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image = image.astype(dtype=np.float64)
    if image.max() == 255.0:
        image = (image - 127.5) / 127.5  # type: ignore
    else:
        image = (image - 0.5) / 0.5  # type: ignore
    return image


def output_to_intensity(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    image = (image + 1.0) / 2.0  # type: ignore
    image = 255.0 * image
    image = image.astype(dtype=np.uint8)
    return image


def output_to_binary(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    image = image.copy()
    image = (image + 1.0) / 2.0  # type: ignore
    image = image > threshold  # type: ignore
    image = image.astype(np.bool)
    return image


def load_real_data(filename):
    data = np.load(filename)
    X1 = data["arr_0"]  # type: ignore
    X2 = data["arr_1"]  # type: ignore
    X3 = data["arr_2"]  # type: ignore

    assert isinstance(X1, np.ndarray)
    assert isinstance(X2, np.ndarray)
    assert isinstance(X3, np.ndarray)

    X1 = intensity_to_input(X1)
    X2 = binary_to_input(X2)
    X3 = binary_to_input(X3)

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


def resize_all(X_realA, X_realB, X_realC, out_shape_space):
    X_realA = resize_stack(data=X_realA, out_shape_space=out_shape_space)
    X_realB = resize_stack(data=X_realB, out_shape_space=out_shape_space)
    X_realC = resize_stack(data=X_realC, out_shape_space=out_shape_space)
    return [X_realA, X_realB, X_realC]


def resize_stack(data, out_shape_space):
    out = []
    for index in range(len(data)):
        im = cv2.resize(
            data[index, ...], dsize=out_shape_space, interpolation=cv2.INTER_LANCZOS4
        )
        out.append(im)
    out = np.stack(out, axis=0)
    return out
