import rl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from numpy.typing import NDArray
from time import time


def plot_values(values: NDArray):
    """
    Plots a heatmap representing the state values in a grid world.

    Parameters:
    - values (NDArray[np.float64]): A 2D numpy array of shape (height, width) where each element
                                    represents the computed value of that state.

    Returns:
    - None: Displays the plot.
    """
    assert values.ndim == 2, f"Expected 2D array of shape (height, width), got shape {values.shape}"
    sns.heatmap(values, annot=True, cbar_kws={"label": "Value"})
    plt.show()


def plot_qvalues(q_values: NDArray, action_symbols: list[str]):
    """
    Plots a heatmap of the maximum Q-values in each state of a grid world and overlays symbols
    to represent the optimal action in each state.

    Parameters:
    - q_values (NDArray): A 3D numpy array of shape (height, width, n_actions), where each cell contains Q-values
                                      for four possible actions (up, down, right, left).
    - env (Labyrinth): The environment instance to access action symbols.

    Returns:
    - None: Displays the plot.
    """
    assert q_values.ndim == 3, f"Expected 3D array of shape (height, width, n_actions), got shape {q_values.shape}"
    assert q_values.shape[-1] == len(action_symbols), "Number of action symbols should match the number of actions"
    height, width = q_values.shape[:2]

    # Calculate the best action and max Q-value for each cell
    best_actions = np.argmax(q_values, axis=2)
    max_q_values = np.max(q_values, axis=2)

    # Plotting the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(max_q_values, origin="upper")
    plt.colorbar(label="Max Q-value")
    # Overlay best action symbols
    for i in range(height):
        for j in range(width):
            action_symbol = action_symbols[best_actions[i, j]]
            plt.text(j, i, action_symbol, ha="center", va="center", color="black", fontsize=12)

    # Labels and layout
    plt.title("Q-value Heatmap with Optimal Actions")
    plt.grid(False)
    plt.show()


def random_moves(env: rl.Labyrinth, n_steps: int) -> None:
    """
    Makes random moves in the environment and renders each step.

    Parameters:
    - env (Labyrinth): The environment instance where random moves will be performed.
    - n_steps (int): Number of random steps to perform.

    Returns:
    - None
    """
    env.reset()
    env.render()
    episode_rewards = 0
    for _ in range(n_steps):
        random_action = random.choice(env.available_actions())
        reward = env.step(random_action)
        episode_rewards += reward
        if env.is_done:
            print("collected reward =", episode_rewards)
            env.reset()
            episode_rewards = 0
        env.render()


def random_seeds(n: int) -> list[int]:
    """
    Generates n random seeds.

    Parameter:
    - n (int): The number of seeds to generate.
    """
    return [random.random() for _ in range(n)]


def run_value_iteration(env: rl.Labyrinth, render: bool) -> None:
    """
    Runs the Value Iteration algorithm.

    Parameter:
    - env (rl.Labyrinth): The environment instance where the algorithm will be executed.
    - render (bool): If the results are to be plotted.
    """
    δ = 0
    γ = 0.9
    algo = rl.ValueIteration(env=env, gamma=γ)
    start = time()
    algo.train(δ)
    end = time()
    delta_t = (end - start) * 1000 # for ms
    print(f"Value Iteration trained with Delta δ = {δ} | Discount value γ = {γ} | In {delta_t} milliseconds.")

    if render:
        plot_values(algo.get_value_table())

    env.reset()
    print()


def run_q_learning(env: rl.Labyrinth, render: bool) -> list[int]:
    """
    Runs the Q-learning algorithm.

    Parameter:
    - env (rl.Labyrinth): The environment instance where the algorithm will be executed.
    - render (bool): If the results are to be plotted.

    Returns:
    - list[int]: A list containing the score values for each episode.
    """

    def run_q_learning_policy(policy: rl.qlearning.Policy):
        γ = 0.9 # gamma
        𝑎 = 0.1 # alpha
        n_steps = 20_000

        algo = rl.QLearning(env=env, gamma=γ, alpha=𝑎, policy=policy)
        start = time()
        scores = algo.train(n_steps)
        end = time()
        delta_t = (end - start) * 1000 # for ms

        print(f"Q-learning trained with Discount value γ = {γ} | Learning rate 𝑎 = {𝑎} | In {delta_t} milliseconds.")
        if isinstance(policy, rl.qlearning.SoftmaxPolicy):
            print(f"Policy used: Softmax with Temperature value 𝜏 = {𝜏}")
        elif isinstance(policy, rl.qlearning.EpsilonGreedyPolicy):
            print(f"Policy used: Epsilon-Greedy with Epsilon value 𝜀 = {𝜀}")
        else:
            raise TypeError("non")

        if render:
            action_to_symbol = ["↑", "↓", "→", "←", "·"]
            plot_qvalues(algo.get_q_table(), action_symbols=action_to_symbol)
        
        print()

    values = {}

    # epsilon greedy policy
    𝜀_values = [0, 0.1, 0.5, 0.9, True]
    for 𝜀 in 𝜀_values:
        policy = rl.qlearning.EpsilonGreedyPolicy(𝜀)
        run_q_learning_policy(policy=policy)
        env.reset()

    # softmax policy
    𝜏_values = [0.01, 1, 10, 10000000, True]
    for 𝜏 in 𝜏_values:
        policy = rl.qlearning.SoftmaxPolicy(𝜏)
        run_q_learning_policy(policy=policy)
        env.reset()


    return values



def main():
    render = True
    # ==================================================== #
    #                        Seed                          #
    # ==================================================== #
    random.seed(0)

    # ==================================================== #
    #             Environment initialisation               #
    # ==================================================== #
    # probability of taking a random action instead of the chosen one
    probability = 0.1
    env = rl.Labyrinth(p=probability)
    env.reset()

    # ==================================================== #
    #                   Value Iteration                    #
    # ==================================================== #
    run_value_iteration(env=env, render=render)

    # ==================================================== #
    #                      Q-learning                      #
    # ==================================================== #
    number_of_seeds = 1
    seeds = random_seeds(number_of_seeds)
    for seed in seeds:
        random.seed(seed)
        values = run_q_learning(env=env, render=render)


if __name__ == "__main__":
    main()

