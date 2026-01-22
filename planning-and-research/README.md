# Project 1: Adversarial Research

## Context

### Technical implementation

Within the context of the Laser Learning Environment (LLE), this project implements three classical research algorithms:
- Depth First Search (DFS)
- Breadth First Search (BFS)
- A* (with Manhattan distance as heuristic function)

Each algorithm is used to solve one kind of problem. Every problem adds some complexity to the previous one. Three types can be found:
- ExitProblem which consists in the agent searching the exit of the environment to exit it;
- GemProblem which consits in the agent collecting all the gems in the environment before exiting it;
- CornerProblem which consists in the agent visiting all 4 corners of the environment before collecting all the gems and finally exiting the environment.

### General use cases

- State-Space Search (Problem Solving): Core AI domain focused on exploring possible states to reach a goal. BFS, DFS, and A* are standard techniques.
- Pathfinding and Navigation: Used to find optimal or feasible paths in graphs or maps. A* is especially common due to heuristics.
- Planning: Computes sequences of actions from an initial state to a goal. Search algorithms explore action trees.
- Game AI: Explores game states to decide moves. Often combined with adversarial search.
- Robotics: Used for motion planning and decision making. Search helps robots navigate and act safely.
- Automated Reasoning: Searches through logical states or proofs. DFS and BFS are commonly used in inference systems.

## Run the project

At the root of this directory:
```python3 src/main.py```

or

```python3 src/main.py --help```

if you wish to execute specific parts of the code.


## Analysis

The ```main.py``` file contains a big chunk of code used to gather the data throughout the execution of the program. Feel free to comment/uncomment it as it suits your needs !