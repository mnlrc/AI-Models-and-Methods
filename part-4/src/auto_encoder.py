from typing import List
import numpy as np
from tqdm import tqdm


class AutoEncoder:
    """
    Represents a 4-layers auto-encoder.
    """

    def __init__(self, input_dim, encoded_dim, learning_rate):
        """
        Initialisation function.
        Parameters:
            - input_dim (int): the shape of the input (flattened)
            - hidden_dim (int): the shape of the encoded vector
            - learning_rate (float): the learning rate factor
        """
        self.mu = learning_rate
        self.W1 = (np.random.random((input_dim, input_dim // 2)) - 0.5) * 0.001
        self.W2 = (np.random.random((input_dim // 2, encoded_dim)) - 0.5) * 0.001
        self.W3 = (np.random.random((encoded_dim, input_dim // 2)) - 0.5) * 0.001
        self.W4 = (np.random.random((input_dim // 2, input_dim)) - 0.5) * 0.001

    def loss(self, x: np.ndarray, y: np.ndarray) -> float: ...

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encodes an input vector x.
        """
        ...

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decodes an encoded vector x.
        """
        ...

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass.
        """
        ...

    def backward(self, x: np.ndarray) -> None:
        """
        Updates the weights of the network using the backpropagation.
        """
        e4 = ...
        d4 = ...

        e3 = ...
        d3 = ...

        e2 = ...
        d2 = ...

        e1 = ...
        d1 = ...

        self.W4 -= 0
        self.W3 -= 0
        self.W2 -= 0
        self.W1 -= 0

    def train(self, x_train: np.ndarray, epochs: int = 10, batch_size: int = 16) -> List[float]:
        """
        Trains the auto-encoder on the given dataset.
        Parameters:
            - x_train (np.ndarray): the dataset containing the input vectors.
            - epochs (int): the number of epochs to train the auto-encoder with.
            - batch_size (int): the size of each training batch.
        Returns:
            - losses (List[int]): the training loss of each epoch.
        """
        losses = []
        for epoch in range(epochs):
            for i in tqdm(range(0, x_train.shape[0], batch_size)):
                self.forward(x_train[i : i + batch_size])
                self.backward(x_train[i : i + batch_size])
            output = self.forward(x_train)
            loss = self.loss(output, x_train)
            losses.append(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.7f}")
        return losses
