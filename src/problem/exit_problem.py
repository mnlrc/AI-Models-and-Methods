from lle import WorldState
from .problem import SearchProblem


class ExitProblem(SearchProblem[WorldState]):
    """
    A simple search problem where the agents must reach the exit **alive**.
    """

    def is_goal_state(self, state: WorldState) -> bool:
        raise NotImplementedError()
