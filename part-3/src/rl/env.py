import random
from itertools import product

from lle import Action, EventType, World, WorldState

BOTTOM_LEFT_EXIT = (6, 0)


class Labyrinth:
    def __init__(self, p: float = 0.0):
        """
        Parameters:
        - p: Probability of taking a random action instead of the chosen one.
        """
        self._p = p
        self._world = World("""
                .  .  .  X  @  .  S0
                .  @  .  .  .  .  @
                .  @  .  .  .  @  .
                .  .  @  .  V  .  .
                @  .  @  .  V  @  .
                .  .  .  .  .  .  .
                X  @  .  V  @  .  .""")

        self._done = False
        self._first_render = True
        all_positions = set(product(range(self._world.height), range(self._world.width)))
        self._valid_positions = list(all_positions - set(self._world.wall_pos) - set(self._world.exit_pos))
        self._unvalid_positions = list(set(self._world.wall_pos) | set(self._world.exit_pos) | set(self._world.void_pos))

    @property
    def map_size(self):
        """The (height, width) of the labyrinth."""
        return (self._world.height, self._world.width)

    def set_state(self, state: tuple[int, int]):
        """
        Set the agent's state (position) in the world.

        Parameters:
        - state (tuple[int, int]): The (row, column) position to place the agent in.
        """
        self._world.set_state(WorldState([state], []))

    @property
    def valid_states(self) -> list[tuple[int, int]]:
        """The list of valid positions for the agent."""
        return self._valid_positions
    
    @property
    def unvalid_states(self) -> list[tuple[int, int]]:
        """
        The list of not valid positions for the agent.
        For printing purposes."""
        return self._unvalid_positions

    def reset(self):
        """
        Reset the labyrinth to its initial state.

        Returns:
        - tuple[int, int]: The initial observation (agent's starting position).
        """
        self._done = False
        self._world.reset()
        return self.agent_position

    def available_actions(self):
        """
        Retrieve all available actions in the current state.
        If an agent is dead, the only available action is `Action.STAY`.
        """
        return self._world.available_actions()[0]

    def step(self, action: Action | int, state: tuple[int, int] | None = None):
        """
        Perform an action in the environment and return the associated reward.
        If a state is provided, set the agent to that state before executing the action.
        If the action is an integer, it is converted to the corresponding Action enum.
        """
        if self._done:
            raise RuntimeError("Cannot step in a finished environment. Call `reset()` first.")
        if state is not None:
            self.set_state(state)
        if isinstance(action, int):
            action = Action(action)
        if random.random() < self._p:
            action = random.choice(self.available_actions())
        events = self._world.step(action)
        reward = 0.0
        for event in events:
            if event.event_type == EventType.AGENT_DIED:
                self._done = True
                return -1.0
            if event.event_type == EventType.AGENT_EXIT:
                self._done = True
                if self.agent_position == BOTTOM_LEFT_EXIT:
                    reward = 10.0
                else:
                    reward = 1.0
        return reward
    
    def deterministic_step(self, action: Action | int, state: tuple[int, int] | None = None):
        """
        Perform an action in the environment without taking into account any probability
        and without modifying the world's internal state (reset at the end to the initial one).
        Return the associated reward and the new state.
        If a state is provided, set the agent to that state before executing the action.
        If the action is an integer, it is converted to the corresponding Action enum.
        """
        current_done = self._done
        current_state = self.agent_position

        if self._done:
            raise RuntimeError("Cannot step in a finished environment. Call `reset()` first.")
        if state is not None:
            self.set_state(state)
        if isinstance(action, int):
            action = Action(action)
        events = self._world.step(action)
        reward = 0.0
        for event in events:
            if event.event_type == EventType.AGENT_DIED:
                self._done = True
                reward -1.0
            if event.event_type == EventType.AGENT_EXIT:
                self._done = True
                if self.agent_position == BOTTOM_LEFT_EXIT:
                    reward = 10.0
                else:
                    reward = 1.0

        next_state = self.agent_position

        # resetting the world state to previous one
        self._done = current_done
        self.set_state(current_state)

        return reward, next_state

    @property
    def agent_position(self) -> tuple[int, int]:
        """
        Get the agent's current position in the labyrinth.

        Returns:
        - tuple[int, int]: The (row, column) position of the agent.
        """
        return self._world.agents_positions[0]

    def get_image(self):
        return self._world.get_image()

    def render(self) -> None:
        """
        Render the labyrinth environment using OpenCV.
        """
        import cv2

        img = self._world.get_image()
        if self._first_render:
            # Solves a bug such that the first rendering is not displayed correctly the first time
            cv2.imshow("Labyrinth", img)
            cv2.waitKey(1)
            self._first_render = False
            import time

            time.sleep(0.1)

        cv2.imshow("Labyrinth", img)
        cv2.waitKey(1)

    @property
    def is_done(self) -> bool:
        return self._done
