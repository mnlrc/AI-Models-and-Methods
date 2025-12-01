import numpy as np
import matplotlib.pyplot as plt


def plot_image(v: np.ndarray):
    """
    Displays a squared image, given a flat vector.
    """
    plt.imshow(np.reshape(v, (int(v.shape[0] ** 0.5), int(v.shape[0] ** 0.5))), cmap="gray")
    plt.show()


def get_dataset(filepath: str):
    """
    Reads and preprocess a training and testing dataset.
    Parameters:
        - filepath (str): the path to the csv file containing the dataset.
    Returns:
        - x_train, y_train, x_test, y_test (np.ndarray): the images and labels for training and testing sets.
    """
    print("Reading dataset...")
    d = np.loadtxt(filepath, delimiter=",", dtype=str)[1:].astype(np.int64)
    x, y = d[:, 1:], d[:, 0].T
    return x / 255.0, y


def activation(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """
    The sigmoid activation function.
    """
    ...


def derivative(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """
    The derivative of the sigmoid activation function.
    """
    ...
