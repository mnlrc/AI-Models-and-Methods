import numpy as np
from .env import Labyrinth
import lle
from copy import deepcopy

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
        for _ in range(50):
        # while True:
            V_new = self.V
            for s in self.V.keys():
                try:
                    self.env.set_state(s)
                except lle.exceptions.InvalidWorldStateError:
                    # not considering states where agent is dead; it can't make any action from there
                    self.env.reset()
                    continue

                actions = self.env.available_actions()
                best_sum = -np.inf

                for a in actions:
                    transitions = self.get_transitions(a)
                    new_sum = 0
                    # print(s)
                    # print(a)
                    for next_s, reward, probability in transitions:
                        # print(probability)
                        # print(reward)
                        new_sum += probability * (reward + self.gamma * self.V[next_s])
                    # print(new_sum)
                    best_sum = max(new_sum, best_sum)

                V_new[s] = best_sum
                print("BEST SUM: ", best_sum)
                print("PREV_VAL: ", self.V[s])
                print(self.V)
                print("NEW_DELTA: ", abs(self.V[s] - V_new[s]))
                print()
                current_delta = max(current_delta, abs(self.V[s] - V_new[s]))

            self.V = V_new
            # print(current_delta)
            # if current_delta < delta:
                # break  


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
    
    def get_transitions(self, action) -> dict[tuple[int, int], float]:
        """
        Retrieve all the attainable states from the agent's current position using available actions.

        Returns:
        - dict[tuple[int, int], float]: A map contaning all the reachable states with the corresponding reward. 
        """
        transitions = []
        actions = self.env.available_actions()

        # probability of taking the wanted action
        main_probability = 1 - self.env._p

        # probability of another action being taken
        else_probability = self.env._p / len(actions)

        reward, s_next, done = self.env.deterministic_step(action)
        # if not done:
        transitions.append((s_next, reward, main_probability))

        for a in actions:
            if a == action:
                continue
            reward, s_next, done = self.env.deterministic_step(a)
            # if done:
                # continue
            transitions.append((s_next, reward, else_probability))

        return transitions
