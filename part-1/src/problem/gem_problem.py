from lle import WorldState
from .problem import SearchProblem


class GemProblem(SearchProblem[WorldState]):
    def is_goal_state(self, state: WorldState) -> bool:
        """
        A search where the agents must collect all gems and reach the exit **alive** once
        all the gems have been collected.
        """

        # if there is no gem, we just need to check if all agents are at the exits
        if not len(state.gems_collected) == 0:
            # checking if all gems have been collected
            for gem in state.gems_collected:
                if not gem:
                    return False
        
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