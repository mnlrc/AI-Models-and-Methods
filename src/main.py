from search import astar, bfs, dfs
from lle import World
from problem import gem_problem
import time
import cv2
import sys
import matplotlib.pyplot as plt


# an easy map with just one gem not to far from the exit
EASY_MAP = """
S0 . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . G . X
"""

# a simple and unique path to get the only gem and go to the exit
ONE_PATH_MAP = """
S0 . . . . @ . . . .
@  @ @ @ . @ . . . .
.  . . @ . @ @ @ @ .
.  . . @ . . . . @ .
.  . . @ @ @ @ . @ @
.  . . . . . @ G . X
"""

# not to complicate map but many gems on the map
MANY_GEMS_MAP = """
S0 . . . G . . . . G
.  G . . . . @ . . .
.  . . . . G . . . G
.  G . . . . . G . .
.  . . . . G . . . .
X  . . G . . L1N G X X
"""

# a complex map with some gems and some walls and the exit on the other side of the map
COMPLEX_MAP = """
S0 . . . . . . . . .
X  . G L1E . . . . @ .
.  . . . . . . G . .
.  . @ . @ . . . . G
@  @ . . . @ . . . G
G  . . @ G @ . . L3N G
"""

IMPOSSIBLE_MAP = """
S0 . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . . @ @
.  . . . . . . G @ X
"""

def show(world: World):
    img = world.get_image()
    cv2.imshow("Visualisation", img)
    cv2.waitKey(1)

def show_solution(solution, world):
    world.reset()
    show(world)
    if solution is not None:
        for action in solution.actions:
            time.sleep(0.2)
            world.step(action)
            show(world)
    else:
        show(world)
        print("NO SOLUTION FOUND !")
        # time.sleep(5)

def test_world(world: World, data: dict):
    """
    Take a world and test the 3 algo DFS, BFS and A*
    """
    problem = gem_problem.GemProblem(world)

    debut = time.time()
    dfs_problem = dfs(problem, data)
    end = time.time()
    delta = end - debut
    data["dfs"]["times"].append(delta)
    print(f"Time to execute DFS: {delta} sec")
    if dfs_problem is not None:
        print(f"Path length of DFS: {dfs_problem.n_steps}")
        data["dfs"]["steps"].append(dfs_problem.n_steps)

    debut = time.time()
    bfs_problem = bfs(problem, data)
    end = time.time()
    delta = end - debut
    data["bfs"]["times"].append(delta)
    print(f"Time to execute BFS: {delta} sec")
    if bfs_problem is not None:
        print(f"Path length of BFS: {bfs_problem.n_steps}")
        data["bfs"]["steps"].append(bfs_problem.n_steps)

    debut = time.time()
    astar_problem = astar(problem, data)
    end = time.time()
    delta = end - debut
    data["astar"]["times"].append(delta)
    print(f"Time to execute A*: {delta} sec")
    if astar_problem is not None:
        print(f"Path length of A*: {astar_problem.n_steps}")
        data["astar"]["steps"].append(astar_problem.n_steps)

    if not "-nogui" in sys.argv:
        show_solution(dfs_problem, world)
        show_solution(bfs_problem, world)
        show_solution(astar_problem, world)

    print()

def main():
    if len(sys.argv) > 3:
        print("Too much arguments have been giver !")
        print("Use `python main.py --help` to see the available commands")
        return
    
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ["-h", "--help", "-help", "--h"]:
        print("Just run the program to start it and and execute all the maps.\n")
        print("Options:\n")
        print("easy_map             Launch only the 'EASY_MAP'")
        print("one_path_map         Launch only the 'ONE_PATH_MAP'")
        print("many_gems_map        Launch only the 'MANY_GEMS_MAP'")
        print("complex_map          Launch only the 'COMPLEX_MAP'")
        print("impossible_map       Launch only the 'IMPOSSIBLE_MAP'")
        print("-nogui               Can be use with any of the command above to just run the algorithms without the GUI to show all the solutions")
        print("\nIf you use a specific map with the -nogui option, enter the name of the map first and then -nogui")

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "easy_map":
        easy_world = World(EASY_MAP)
        print("Launching test for the EASY MAP :")
        test_world(easy_world)

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "one_path_map":
        one_path_world = World(ONE_PATH_MAP)
        print("Launching test for the ONE PATH MAP :")
        test_world(one_path_world)

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "many_gems_map":
        many_gems_world = World(MANY_GEMS_MAP)
        print("Launching test for the MANY GEMS MAP :")
        test_world(many_gems_world)

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "complex_map":
        complex_world = World(COMPLEX_MAP)
        print("Launching test for the COMPLEX MAP :")
        test_world(complex_world)

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "impossible_map":
        impossible_world = World(IMPOSSIBLE_MAP)
        print("Launching test for the IMPOSSIBLE MAP")
        test_world(impossible_world)

    elif len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1].lower() == "-nogui"):
        map0_data = {
            "dfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "bfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "astar": {
                "times": [],
                "steps": [],
                "nodes": []
            }
        }
        map1_data = {
            "dfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "bfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "astar": {
                "times": [],
                "steps": [],
                "nodes": []
            }
        }
        map2_data = {
            "dfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "bfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "astar": {
                "times": [],
                "steps": [],
                "nodes": []
            }
        }
        map3_data = {
            "dfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "bfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "astar": {
                "times": [],
                "steps": [],
                "nodes": []
            }
        }
        map4_data = {
            "dfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "bfs": {
                "times": [],
                "steps": [],
                "nodes": []
            },
            "astar": {
                "times": [],
                "steps": [],
                "nodes": []
            }
        }
        for _ in range(50):
            easy_world = World(EASY_MAP)
            one_path_world = World(ONE_PATH_MAP)
            many_gems_world = World(MANY_GEMS_MAP)
            complex_world = World(COMPLEX_MAP)
            impossible_world = World(IMPOSSIBLE_MAP)

            print("Launching test for the EASY MAP :")
            test_world(easy_world, map0_data)
            print("Launching test for the ONE PATH MAP :")
            test_world(one_path_world, map1_data)
            print("Launching test for the MANY GEMS MAP :")
            test_world(many_gems_world, map2_data)
            print("Launching test for the COMPLEX MAP :")
            test_world(complex_world, map3_data)
            print("Launching test for the IMPOSSIBLE MAP")
            test_world(impossible_world, map4_data)

        print("MAP0: ", map0_data)
        print()
        print()
        print("MAP1: ", map1_data)
        print()
        print()
        print("MAP2: ", map2_data)
        print()
        print()
        print("MAP3: ", map3_data)
        print()
        print()
        print("MAP4: ", map4_data)
        print()
        print()

    else:
        print("Invalid argument !")
        print("Use `python main.py --help` to see the available commands")
        return

if __name__ == "__main__":        
    main()