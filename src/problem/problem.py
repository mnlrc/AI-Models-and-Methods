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
        
        new_state = self.world.set_state(state)
        available_actions = self.world.available_joint_actions()
        self.world.set_state(state)
        previous_state = self.world.get_state()

        for agent_actions in available_actions:
                try:
                    events = self.world.step(agent_actions)
                    # if an agent dies, the game is over thus not considering this state as a successor
                    if EventType.AGENT_DIED in events:
                        continue
                    new_state = self.world.get_state()
                    successors.append((new_state, agent_actions[0]))

                    self.world.set_state(previous_state) # reseting to the previous state
                except exceptions.InvalidActionError:
                    continue

        return successors

    def __manhattan_distance(self, agent: tuple[int, int], exit: tuple[int, int]) -> float:
        """
        Simple private method to calculate manhattan distance based on two points.
        """
        agent_x, agent_y = agent
        exit_x, exit_y = exit
        return abs(agent_x - exit_x) + abs(agent_y - exit_y)

    def heuristic(self, problem_state: S) -> float:
        """
        Calculates the heuristic for the A* search algorithm.
        """
        heuristic = 0
        if len(problem_state.agents_positions) > len(self.world.exit_pos):
            raise ValueError("There aren't enough exits for all the agents")
        
        if not problem_state.agents_positions == self.world.exit_pos: # checking if all agent arrived to their destination
            for agent_pos in problem_state.agents_positions:
                best = float("+inf")
                for exit in self.world.exit_pos:
                    temp = self.__manhattan_distance(agent_pos, exit)

                    if temp < best:
                        best = temp

                heuristic += best
        
        return heuristic