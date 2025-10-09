from search import astar, bfs, dfs
from lle import World
from problem import gem_problem
import time
import cv2
import sys


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
S0 . S1 . G . . . . G
.  G . . . . . . . .
.  . . . . G . . . G
.  G . . . . . G . .
.  . . . . G . . . .
.  . . G . . . G X X
"""

# a complex map with some gems and some walls and the exit on the other side of the map
COMPLEX_MAP = """
S0 . . . . G . . @ G
.  @ . . . . @ . . .
G  @ . . . @ G @ @ .
@  . . G @ . . G @ G
.  . . . . . @ . . @
G  @ G @ . . . G . X
"""

NOT_POSSIBLE_MAP = """
S0 . @ . . . . . . .
.  . @ . . . . . . .
@  @ @ . . . . . . .
.  . . . . . . . . .
.  . . . . . . . . .
.  . . . . . . G . X
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
        print("NO POSSIBLE SOLUTION FOUND !")
        time.sleep(5)

def test_world(world: World):
    """
    Take a world and test the 3 algo DFS, BFS and A*
    """
    problem = gem_problem.GemProblem(world)

    debut = time.time()
    dfs_problem = dfs(problem)
    end = time.time()
    print(f"Time to execute DFS: {end - debut} sec")
    if dfs_problem is not None:
        print(f"Path length of DFS: {dfs_problem.n_steps}")

    debut = time.time()
    bfs_problem = bfs(problem)
    end = time.time()
    print(f"Time to execute BFS: {end - debut} sec")
    if bfs_problem is not None:
        print(f"Path length of BFS: {bfs_problem.n_steps}")

    debut = time.time()
    astar_problem = astar(problem)
    end = time.time()
    print(f"Time to execute A*: {end - debut} sec")
    if astar_problem is not None:
        print(f"Path length of A*: {astar_problem.n_steps}")

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
        print("not_possible_map     Launch only the 'NOT_POSSIBLE_MAP'")
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

    elif len(sys.argv) > 1 and sys.argv[1].lower() == "not_possible_map":
        not_possible_world = World(NOT_POSSIBLE_MAP)
        print("Launching test for the NOT POSSIBLE MAP")
        test_world(not_possible_world)

    elif len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1].lower() == "-nogui"):
        easy_world = World(EASY_MAP)
        one_path_world = World(ONE_PATH_MAP)
        many_gems_world = World(MANY_GEMS_MAP)
        complex_world = World(COMPLEX_MAP)
        not_possible_world = World(NOT_POSSIBLE_MAP)

        print("Launching test for the EASY MAP :")
        test_world(easy_world)
        print("Launching test for the ONE PATH MAP :")
        test_world(one_path_world)
        print("Launching test for the MANY GEMS MAP :")
        test_world(many_gems_world)
        print("Launching test for the COMPLEX MAP :")
        test_world(complex_world)
        print("Launching test for the NOT POSSIBLE MAP")
        test_world(not_possible_world)

    else:
        print("Invalid argument !")
        print("Use `python main.py --help` to see the available commands")
        return

if __name__ == "__main__":        
    main()