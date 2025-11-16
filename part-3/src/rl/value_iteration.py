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
        positions = [self.env.agent_position]
        while True:
            for agent_pos in positions:
                new_positions = []
                v = self.V[agent_pos] # holding current state's value
                best = - np.inf
                next_states = self.get_next_states(self.env.available_actions())

                for s_next, reward in next_states.items():
                    new_positions.append(s_next)
                    best += 0.5 * (reward + self.gamma * self.V[s_next])

                if (best > self.V[agent_pos]):
                    self.V[agent_pos] = best

                current_delta = max(current_delta, abs(self.V[agent_pos] - v))
            positions = new_positions
            new_positions.clear()

            if current_delta < delta:
                break


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