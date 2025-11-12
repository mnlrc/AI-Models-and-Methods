import numpy as np
from .env import Labyrinth


class ValueIteration:
    """
    Value Iteration algorithm for solving a reinforcement learning environment.
    The algorithm iteratively updates the estimated values of states to find an optimal policy.
    """

    def __init__(self, env: Labyrinth, gamma: float):
        assert 0 < gamma <= 1, "Gamma must be in (0, 1]"
        self.env = env
        self.gamma = gamma

    def train(self, delta: float):
        """
        Train the agent using value iteration until the value function converges.

        Parameters:
        - delta (float): The threshold for convergence.
        """
        raise NotImplementedError()

    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        raise NotImplementedError()
