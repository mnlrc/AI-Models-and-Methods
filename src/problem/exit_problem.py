from lle import WorldState
from .problem import SearchProblem


class ExitProblem(SearchProblem[WorldState]):
    """
    A simple search problem where the agents must reach the exit **alive**.
    """

    def is_goal_state(self, state: WorldState) -> bool:
        if not any(state.agents_alive):
            return False
        tem_agents_pos = state.agents_positions.copy()
        
        for exit in self.world.exit_pos:
            for agent_position in state.agents_positions:
                if exit == agent_position:
                    tem_agents_pos.remove(agent_position)

        return len(tem_agents_pos) == 0