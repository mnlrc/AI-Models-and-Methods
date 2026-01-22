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


def dfs(problem: SearchProblem, data: dict, data_key: str) -> Optional[Solution]:
    visited_nodes = 0
    stack = []
    visited = set()
    
    first_node = SearchNode(problem.initial_state, None, None)
    stack.append(first_node)
    visited.add(first_node.state)
    
    while stack:
        current_node = stack.pop()
        visited_nodes += 1

        if problem.is_goal_state(current_node.state):
            data[data_key]["dfs"].append(visited_nodes)
            return Solution.from_node(current_node)
        
        current_successors = problem.get_successors(current_node.state)
        for successor_state, action in reversed(current_successors):
            if successor_state not in visited:
                new_node = SearchNode(successor_state, current_node, action)
                stack.append(new_node)
                visited.add(successor_state)

    data[data_key]["dfs"].append(visited_nodes)


def bfs(problem: SearchProblem, data: dict, data_key: str) -> Optional[Solution]:
    visited_nodes = 0
    queue = PriorityQueue()
    visited = set()

    first_node = SearchNode(problem.initial_state, None, None)
    queue.push(first_node, 0)
    visited.add(first_node.state)

    while not queue.is_empty():
        current_node = queue.pop()
        visited_nodes += 1

        if problem.is_goal_state(current_node.state):
            data[data_key]["bfs"].append(visited_nodes)
            return Solution.from_node(current_node)

        current_successors = problem.get_successors(current_node.state)
        for successor_state, action in reversed(current_successors):
            if successor_state not in visited:
                new_node = SearchNode(successor_state, current_node, action)
                queue.push(new_node, 0)
                visited.add(successor_state)

    data[data_key]["bfs"].append(visited_nodes)

def astar(problem: SearchProblem, data: dict, data_key: str) -> Optional[Solution]:
    visited_nodes = 0
    queue = PriorityQueue()
    visited = set()

    first_node = SearchNode(problem.initial_state, None, None, 0)
    queue.push(first_node, first_node.cost) # cost of the first Node is always 0
    visited.add(first_node.state)

    while not queue.is_empty():
        current_node = queue.pop()
        visited_nodes += 1

        if problem.is_goal_state(current_node.state):
            data[data_key]["astar"].append(visited_nodes)
            return Solution.from_node(current_node)
        
        current_successors = problem.get_successors(current_node.state)
        for successor_state, action in current_successors:
            if successor_state not in visited:
                new_node = SearchNode(successor_state, current_node, action, current_node.cost + 1)
                new_cost = current_node.cost + problem.heuristic(successor_state)
                queue.push(new_node, new_cost)
                visited.add(successor_state)

    data[data_key]["astar"].append(visited_nodes)
