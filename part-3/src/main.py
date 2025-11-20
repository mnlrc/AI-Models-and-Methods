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


def main():
    # ==================================================== #
    #                        Seed                          #
    # ==================================================== #
    random.seed(0)
    # np.random.seed(0) # Si vous utilisez numpy pour des tirages aléatoires

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
    # Uncomment for Value Iteration

    # δ = 0.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
    # γ = 0.9
    # algo = rl.ValueIteration(env, 0.9)
    # start = time()
    # algo.train(δ)
    # end = time()
    # delta_t = (end - start) * 1000 # for ms
    # print(f"Value Iteration trained with delta δ = {δ} | discount value γ = {γ} | in {delta_t}ms")
    # plot_values(algo.get_value_table())


    # ==================================================== #
    #                      Q-learning                      #
    # ==================================================== #
    # Uncomment for Q-learning
    policy = rl.qlearning.SoftmaxPolicy(100)
    algo = rl.QLearning(env, 0.9, 0.1, policy)
    algo.train(10_000)
    action_to_symbol = ["↑", "↓", "→", "←", "·"]
    plot_qvalues(algo.get_q_table(), action_symbols=action_to_symbol)

if __name__ == "__main__":
    main()

