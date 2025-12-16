from typing import List
import numpy as np
from tqdm import tqdm
from utils import activation
from utils import derivative


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

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error.
        """
        return np.mean(np.square(np.subtract(x, y)))

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Encodes an input vector x.
        """
        # using matricial product
        x1 = activation(x=np.matmul(x, self.W1))

        # using matricial product
        xhat = activation(x=np.matmul(x1, self.W2))

        return x1, xhat

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decodes an encoded vector x.
        """
        # using matricial product
        x2 = activation(x=np.matmul(x,self.W3))

        # using matricial product
        y = activation(x=np.matmul(x2, self.W4))

        return x2, y

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass.
        """
        x1, xhat = self.encode(x=x)
        x2, y = self.decode(x=xhat)
        return y


    def backward(self, x: np.ndarray) -> None:
        """
        Updates the weights of the network using the backpropagation.
        """
        x1, xhat = self.encode(x=x)
        x2, y = self.decode(x=xhat)

        # Step 1: Calculate output's gradient error
        e4 = np.multiply(2, np.subtract(y, x))
        d4 = np.multiply(e4, derivative(x=y))

        # Step 2: Propagation of errors to layers underneath
        e3 = np.matmul(d4, 
                       np.transpose(self.W4))
        d3 = np.multiply(e3, derivative(x=x2))

        e2 = np.matmul(d3, 
                       np.transpose(self.W3))
        d2 = np.multiply(e2, derivative(x=xhat))

        e1 = np.matmul(d2, 
                       np.transpose(self.W2))
        d1 = np.multiply(e1, derivative(x=x1))

        # Step 3: Updating weights
        self.W4 -= np.multiply(self.mu, 
                               np.matmul(np.transpose(x2), d4))

        self.W3 -= np.multiply(self.mu, 
                               np.matmul(np.transpose(xhat), d3))

        self.W2 -= np.multiply(self.mu, 
                               np.matmul(np.transpose(x1), d2))
        
        self.W1 -= np.multiply(self.mu, 
                               np.matmul(np.transpose(x), d1))
        

    def train(self, x_train: np.ndarray, x_test: np.ndarray, epochs: int = 10, batch_size: int = 16) -> tuple[List[float], np.ndarray, List[float]]:
        """
        Trains the auto-encoder on the given dataset and testing the auto-encoder on the test dataset for each epoch
        while recording the training and testing losses.
        Parameters:
            - x_train (np.ndarray): the dataset containing the input vectors.
            - x_test (np.ndarray): the dataset containing the testing input vectors.
            - epochs (int): the number of epochs to train the auto-encoder with.
            - batch_size (int): the size of each training batch.
        Returns:
            - train_losses (list[float]): the list of training losses recorded at each epoch.
            - test_output (np.ndarray): the reconstructed test output after the final epoch; after the training is complete.
            - test_losses (list[float]): the list of testing losses recorded at each epoch.
        """
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            for i in tqdm(range(0, x_train.shape[0], batch_size)):
                self.forward(x_train[i : i + batch_size])
                self.backward(x_train[i : i + batch_size])
                
            train_output = self.forward(x_train)
            train_loss = self.loss(train_output, x_train)
            train_losses.append(train_loss)

            test_output = self.forward(x_test)
            test_loss = self.loss(test_output, x_test)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Training loss: {train_loss:.7f}, Testing loss: {test_loss:.7f}")
        return train_losses, test_output, test_losses
