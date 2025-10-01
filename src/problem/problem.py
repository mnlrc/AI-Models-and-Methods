from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from lle import World, Action, WorldState, EventType


S = TypeVar("S", bound=WorldState)


class SearchProblem(ABC, Generic[S]):
    """
    A Search Problem is a problem that can be solved by a search algorithm.

    The generic parameter S is the type of the problem state.
    """

    def __init__(self, world: World):
        self.world = world
        world.reset()
        self.initial_state = world.get_state()

    @abstractmethod
    def is_goal_state(self, problem_state: S) -> bool:
        """Whether the given state is the goal state"""

    def get_successors(self, state: S) -> list[tuple[WorldState, Action]]:
        """
        Returns  all possible states that can be reached from the given world state.

        Note that if an agent dies, the game is over and there is no successor to that state.
        """
        successors = []

        # if an agent is dead, the game is over and, therefore, aren't any successors
        if not any(state.agents_alive):
            return successors
        
        available_actions = self.world.available_actions

        for action in available_actions:
            event = self.world.step(action)
            if event == EventType.AGENT_DIED: # if an agent dies, the game is over
                pass
            new_state = self.world.get_state()
            successors.append((new_state, action))

            self.world.set_state(state) # reseting to the current state


    def heuristic(self, problem_state: S) -> float:
        raise NotImplementedError()
