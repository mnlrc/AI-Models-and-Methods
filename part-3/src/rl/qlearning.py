import numpy as np
import numpy.typing as npt
import lle
import random
from .env import Labyrinth


class Policy:
    def select_action(self, qvalues: npt.NDArray[np.float32]) -> int:
        raise TypeError("Select a specific policy")


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float):
        super().__init__()
        assert 0 <= epsilon <= 1, "Epsilon must be in [0, 1]"
        self.epsilon = epsilon

    def select_action(self, qvalues: npt.NDArray[np.float32], current_state: tuple[int, int], 
                      available_actions: list[lle.Action]) -> int:
        # print("EPSILON GREEDY POLICY")
        r = random.random()
        if r <= self.epsilon:
            # random action
            action = random.choice(available_actions).value
        else:
            # picking the action with the best Q-value
            q_state_values = qvalues[current_state]
            action = max(available_actions, key=lambda a: q_state_values[a.value]).value

        return action

class SoftmaxPolicy(Policy):
    def __init__(self, temperature: float):
        super().__init__()
        assert temperature > 0, "Temperature must be positive"
        self.temperature = temperature

    def select_action(self, qvalues: npt.NDArray[np.float32], current_state: tuple[int, int],
                       available_actions: list[lle.Action]) -> int:
        print("SOFTMAX POLICY ACTION CHOICE")
        sum = 0
        h, w = current_state
        for a in available_actions:
            a_int = a.value
            sum += np.e ** (qvalues[h, w, a_int] / self.temperature)

        # contains an associated probability for each action
        probabilities = []
        print(probabilities)
        for a in available_actions:
            a_int = a.value
            num = np.e ** (qvalues[h, w, a_int] / self.temperature)
            probabilities.append(num / sum)

        print(probabilities)
        action = random.choices(population=available_actions, weights=probabilities, k=1)[0]
        return action.value



class QLearning:
    def __init__(self, env: Labyrinth, gamma: float, alpha: float, policy: Policy):
        assert 0 < gamma <= 1, "Gamma must be in (0, 1]"
        assert 0 < alpha <= 1, "Alpha must be in (0, 1]"
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.policy = policy
        # action values in integer:
        # - North = 0
        # - South = 1
        # - East = 2
        # - West = 3
        # - Stay = 4
        h, w = self.env.map_size
        self.q_table = np.zeros(shape=(h, w, 5))

    def get_q_table(self) -> npt.NDArray[np.float32]:
        """
        Retrieve the Q-table as a 3D numpy array for visualization.

        Returns:
        - np.ndarray: A 3D array representing Q-values for each state-action pair.
        """
        return self.q_table

    def train(self, n_steps: int):
        """
        Train the Q-learning agent over a specified number of steps.

        Parameters:
        - n_steps (int): Total number of steps for training.
        """
        counter = 0
        values = []
        for i in range(n_steps):
            counter += 1
            if self.env.is_done:
                print(f"Agent has taken {counter} steps before dying")
                values.append(counter)
                counter = 0
                self.env.reset()

            current_state = self.env.agent_position        
            actions = self.env.available_actions()
            # print(s)
            # print(self.env.agent_position)
            # print(actions)

            # by the policy
            chosen_action = self.policy.select_action(self.q_table, current_state, actions)

            reward = self.env.step(chosen_action, current_state)
            s_next = self.env.agent_position
            h_next, w_next = s_next
            s_next_value = np.max(self.q_table[h_next, w_next])

            h, w = current_state
            # print(self.q_table[current_state, chosen_action])
            # print()
            # print(self.q_table[h, w, chosen_action])
            self.q_table[h, w, chosen_action] = (1 - self.alpha) \
                * self.q_table[h, w, chosen_action] \
                + self.alpha * (reward + self.gamma * s_next_value)
            

            print(f"\rIterations: {i + 1} | \
                  ", end="", flush=True)
                #   Q-Values:\n {self.q_table} \
        
        print()
        print(f"Average steps taken by the agent before dying: {sum(values) / len(values)}")