from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from lle import World, Action, WorldState, EventType, exceptions


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
        
        available_actions = self.world.available_actions()
        self.world.set_state(state)
        previous_state = self.world.get_state()

        for agent_actions in available_actions:
            for action in agent_actions:
                try:
                    events = self.world.step(action)
                    # if an agent dies, the game is over thus not considering this state as a successor
                    if EventType.AGENT_DIED in events:
                        continue
                    new_state = self.world.get_state()
                    successors.append((new_state, action))

                    self.world.set_state(previous_state) # reseting to the previous state
                except exceptions.InvalidActionError:
                    continue

        return successors


    def heuristic(self, problem_state: S) -> float:
        heuristic = 0
        if len(problem_state.agents_positions) > len(self.world.exit_pos):
            raise ValueError("There aren't enough exits for all the agents")
        
        if not problem_state.agents_positions == self.world.exit_pos: # checking if all agent arrived to their destination
            for exit in self.world.exit_pos:
                best = float("+inf")
                for agent_pos in problem_state.agents_positions:
                    agent_x, agent_y = agent_pos
                    exit_x, exit_y = exit
                    temp = abs(agent_x - exit_x) + abs(agent_y - exit_y)

                    if temp < best:
                        best = temp

                heuristic += best
        
        return heuristic