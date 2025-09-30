from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from lle import Action, WorldState

from priority_queue import PriorityQueue
from problem import SearchProblem

S = TypeVar("S", bound=WorldState)


@dataclass
class Solution(Generic[S]):
    actions: list[Action]
    states: list[S]

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    @staticmethod
    def from_node(node: "SearchNode") -> "Solution[S]":
        actions = []
        states = []
        while node.parent is not None:
            actions.append(node.prev_action)
            states.append(node.state)
            node = node.parent
        actions.reverse()
        return Solution(actions, states)


@dataclass
class SearchNode:
    state: WorldState
    parent: Optional["SearchNode"]
    prev_action: Optional[Action]
    cost: float = 0.0

    def __hash__(self) -> int:
        return hash((self.state, self.cost))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchNode):
            return NotImplemented
        return self.state == other.state and self.cost == other.cost


def dfs(problem: SearchProblem) -> Optional[Solution]:
    raise NotImplementedError()


def bfs(problem: SearchProblem) -> Optional[Solution]:
    raise NotImplementedError()


def astar(problem: SearchProblem) -> Optional[Solution]:
    raise NotImplementedError()
