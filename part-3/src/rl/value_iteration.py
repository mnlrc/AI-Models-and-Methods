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
        # setting up states and state values (all zero when starting)
        self.V = {s: 0 for s in self.env.valid_states}

    def train(self, delta: float):
        """
        Train the agent using value iteration until the value function converges.

        Parameters:
        - delta (float): The threshold for convergence.
        """

        # difference of values of two succeeding states
        current_delta = 0
        available_states = [self.env.agent_position]
        probability = 1 - self.env._p
        while True:

            for s in available_states:
                actions = self.env.available_actions()
                best_sum = -np.inf

                for a in actions:
                    
                    next_states = self.get_next_states(a)

                    new_sum = 0.0
                    reward = self.env.step(a)

                    for s_next, prob in next_states.items():
                        new_sum += prob * (reward + self.gamma * self.V[s_next])

                    best_sum = max(best_sum, new_sum)

                self.V_new[s] = best_sum


    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        height, width = self.env.map_size
        results = np.zeros((height, width), dtype=np.float64)
        for pos, value in self.V.items():
            results[pos] = value

        return results
    
    def get_next_states(self, actions) -> dict[tuple[int, int], float]:
        rewards = {}
        for action in actions:
            try:
                reward = self.env.step(action, self.env.agent_position)
                rewards[self.env.agent_position] = reward
            except RuntimeError:
                self.env.reset()
                reward = self.env.step(action, self.env.agent_position)
                rewards[self.env.agent_position] = reward

        return rewards