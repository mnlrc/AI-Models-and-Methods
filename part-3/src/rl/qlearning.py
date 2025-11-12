import numpy as np
import numpy.typing as npt

from .env import Labyrinth


class Policy:
    def select_action(self, qvalues: npt.NDArray[np.float32]) -> int:
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float):
        super().__init__()
        assert 0 <= epsilon <= 1, "Epsilon must be in [0, 1]"
        self.epsilon = epsilon


class SoftmaxPolicy(Policy):
    def __init__(self, temperature: float):
        super().__init__()
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature


class QLearning:
    def __init__(self, env: Labyrinth, gamma: float, alpha: float, policy: Policy):
        assert 0 < gamma <= 1, "Gamma must be in (0, 1]"
        assert 0 < alpha <= 1, "Alpha must be in (0, 1]"
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.policy = policy

    def get_q_table(self) -> npt.NDArray[np.float32]:
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """
        raise NotImplementedError()

    def train(self, n_steps: int):
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        """
        raise NotImplementedError()
