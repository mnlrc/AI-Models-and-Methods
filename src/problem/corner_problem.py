from lle import WorldState, World
from .problem import SearchProblem


class CornerState(WorldState):
    def __init__(
        self,
        agents_positions: list[tuple[int, int]],
        gems_collected: list[bool],
        agents_alive: list[bool],
        corner_positions: set[tuple[int, int]],
        visited_corners: set[tuple[int, int]],
    ):
        super().__init__(agents_positions, gems_collected, agents_alive)
        self.corner_positions = corner_positions
        self.visited_corners = visited_corners

    # from the lle library documentation
    def __new__(
            cls, 
            agents_positions: list[tuple[int, int]], 
            gems_collected: list[bool], 
            agents_alive: list[bool], 
            *args, 
            **kwargs
        ):
        instance = super().__new__(cls, agents_positions, gems_collected, agents_alive)
        return instance

    def __hash__(self) -> int:
        return hash(self.corner_positions, self.visited_corners)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CornerState):
            return TypeError("Wrong type given to compare")
        return self.corner_positions == other.corner_positions and self.visited_corners == other.visited_corners

class CornerProblem(SearchProblem[CornerState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.corners = [(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)]
        # self.initial_state = CornerState(...)

    def is_goal_state(self, state: CornerState) -> bool:
        # 2 steps:
        # 1. -> check corners (if they all have been visited)
        # 2. -> check exits, basically same code than before

        # 1. step
        

        # 2. step
        if not any(state.agents_alive):
            return False
        tem_agents_pos = state.agents_positions.copy()
        
        # checking if agents are on the exits
        for exit in self.world.exit_pos:
            for agent_position in state.agents_positions:
                if exit == agent_position:
                    tem_agents_pos.remove(agent_position)

        # all agents reached an exit and all gems have been collected
        return len(tem_agents_pos) == 0 

    def get_successors(self, state: CornerState):
        successors = []
        for world_state, actions in super().get_successors(state):
            # You probably need to do something like this here
            # next_state = CornerState(...)
            raise NotImplementedError()
        return successors
