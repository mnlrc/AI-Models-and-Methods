import numpy as np
from .env import Labyrinth
import lle
from copy import deepcopy
import main
import time

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
        self.V = np.zeros(shape=self.env.map_size)

    def train(self, delta: float):
        """
        Train the agent using value iteration until the value function converges.

        Parameters:
        - delta (float): The threshold for convergence.
        """

        smallest_variation = np.inf
        iterations = 0
        while True:
            # uncomment if you wish to see the evolution
            # main.plot_values(self.get_value_table())
            variation = 0
            V_next = self.V.copy()
            for s in self.env.valid_states:
                try:
                    self.env.set_state(s)
                except lle.exceptions.InvalidWorldStateError:
                    # not considering invalid states
                    self.env.reset()
                    continue

                actions = self.env.available_actions()
                best_sum = -np.inf

                for a in actions:
                    transitions = self.get_transitions(a)
                    new_sum = 0
                    for next_s, reward, probability in transitions:
                        new_sum += probability * (reward + self.gamma * self.V[next_s])
                    
                    best_sum = max(new_sum, best_sum)

                V_next[s] = best_sum
            
            variation = np.max(np.abs(self.V - V_next))
            self.V = V_next

            if variation <= delta:
                print()
                break

            # for printing purposes
            iterations += 1
            if variation < smallest_variation:
                smallest_variation = variation
            print(
                f"\rIterations: {iterations} | Δ: {smallest_variation:.4f} (δ: {delta})",
                end="",
                flush=True,
            )



    def get_value_table(self) -> np.ndarray:
        """
        Retrieve the current value table as a 2D numpy array.

        Returns:
        - np.ndarray: A 2D array representing the estimated values for each state.
        """
        return self.V
    
    def get_transitions(self, action) -> list[tuple[tuple[int, int], float, float]]:
        """
        Retrieve all the attainable states from the agent's current position using available actions
        without changing the environment.

        Returns:
        - list[tuple[tuple[int, int], float, float]]: A map contaning all the reachable states 
        with the corresponding reward and probability.
        """
        transitions = []
        actions = self.env.available_actions()

        # probability of taking the wanted action
        main_probability = 1 - self.env._p

        # probability of another action being taken
        else_probability = self.env._p / len(actions)

        reward, s_next = self.env.deterministic_step(action)
        transitions.append((s_next, reward, main_probability))

        for a in actions:
            if a == action:
                continue
            reward, s_next = self.env.deterministic_step(a)
            transitions.append((s_next, reward, else_probability))

        return transitions
