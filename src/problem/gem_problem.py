from lle import WorldState
from .problem import SearchProblem


class GemProblem(SearchProblem[WorldState]):
    def is_goal_state(self, state: WorldState) -> bool:
        raise NotImplementedError()
