import numpy as np
import itertools

def distance(cell: tuple[int, int], gem_position: tuple[int, int]):
    """Calculate Euclidean distance between the current cell and the gem position."""
    delta_x = cell[0] - gem_position[0]
    delta_y = cell[1] - gem_position[1]
    return np.sqrt((
        (delta_x ** 2) + 
        (delta_y ** 2)))

def vector_distance(vector1: list[int], vector2: list[2]):
    manhatan_distance = 0

    if len(vector1) != len(vector2):
        raise Exception("The vectors must be the same length")
    
    for i in range(len(vector1)):
        manhatan_distance += np.abs(vector1[i] - vector2[i])

    return manhatan_distance

class BayesianNetwork:
    def __init__(self, grid_size: int, n_gems: int):
        self.grid_size = grid_size
        self.n_gems = n_gems
        self.G = np.ones((grid_size, grid_size)) / (grid_size * grid_size)
    
    def likelihood(self, current_cell: tuple[int, int], distances: list[int], gem_positions: list[tuple[int, int]]):
        """Compute likelihood of observing given distances, given gem positions."""
        𝜆 = 1 # hyperparameter set arbitrary (for now at least)
        observations = []

        for gem_pos in gem_positions:
            observations.append(distance(current_cell, gem_pos))

        manhatan_distance = vector_distance(distances, observations)
        return np.exp(- manhatan_distance / 𝜆)

    
    def infer(self, cell, distances):
        """Update beliefs using inference by enumeration over all possible gem positions."""
        posterior = np.zeros((self.grid_size, self.grid_size))
        
        # ...
        # TO DO
        
        self.G = posterior
    
    def get_belief_distribution(self):
        """Return current belief distribution (posterior)."""
        return self.G
