from lle import WorldState
from .problem import SearchProblem


class ExitProblem(SearchProblem[WorldState]):
    """
    A simple search problem where the agents must reach the exit **alive**.
    """

    def is_goal_state(self, state: WorldState) -> bool:
        if not any(state.agents_alive):
            return False
        exits = self.world.exit_pos

        for i in range(len(exits)):
            if (exits[i] != state.agents_positions[i]):
                return False
        return True

        # for exit in exits:
        #     for agent_position in state.agents_positions:
        #         if exit == agent_position:
        #             exits.remove(exit)
        #             state.agents_positions.remove(agent_position)


        # return len(exits) == 0 and len(state.agents_positions) == 0