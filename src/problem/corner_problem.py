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
            return hash((super().__hash__(), tuple(self.visited_corners)))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CornerState):
            return TypeError("Wrong type given to compare")
        return self.corner_positions == other.corner_positions and self.visited_corners == other.visited_corners

class CornerProblem(SearchProblem[CornerState]):
    def __init__(self, world: World):
        super().__init__(world)
        self.corners = [(0, 0), (0, world.width - 1), (world.height - 1, 0), (world.height - 1, world.width - 1)]
        initial_world_state = world.get_state()
        self.initial_state = CornerState(
            initial_world_state.agents_positions,
            initial_world_state.gems_collected,
            initial_world_state.agents_alive,
            self.corners,
            set()
        )

    def is_goal_state(self, state: CornerState) -> bool:
        # 2 steps:
        # 1st -> check corners (if they all have been visited)
        # 2nd -> check exits, basically same code than before

        # 1st step
        # if all the exits haven't been visited, there is no use in checking the exits

        if not isinstance(state, CornerState):
            return False 

        if not state.corner_positions == state.visited_corners:
            return False
        
        # 2nd step
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

        if not isinstance(state, CornerState):
            raise TypeError("This method only accepts CornerState type")

        for world_state, actions in super().get_successors(state):
            next_state = CornerState(
                world_state.agents_positions,
                world_state.gems_collected,
                world_state.agents_alive,
                self.corners,
                state.visited_corners.copy()
            )
            successors.append((next_state, actions))
        return successors
