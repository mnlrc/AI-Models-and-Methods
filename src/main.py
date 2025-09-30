import cv2
from lle import World, Action


def show(world: World):
    img = world.get_image()
    cv2.imshow("Visualisation", img)
    cv2.waitKey(1)


world = World.level(1)
world.reset()
show(world)

# LEVEL 1
path = [Action.SOUTH] * 5
path += [Action.EAST] * 3
path += [Action.SOUTH] * 5
path += [Action.WEST] * 3
print(path)

for action in path:
    events = world.step(action)
    print(events)
    show(world)
    input("Appuyez sur 'enter' pour passer à l'action suivante...")


# LEVEL 2
# path = [Action.SOUTH, Action.SOUTH] * 5
# path += [Action.EAST, Action.NORTH] * 3
# path += [Action.SOUTH, Action.EAST] * 5
# path += [Action.WEST, Action.WEST] * 3
# print(path)
# 
# for i in range(0, len(path), 2):
    # action = [path[i], path[i + 1]]
    # events = world.step(action)
    # print(events)
    # show(world)
    # input("Appuyez sur 'enter' pour passer à l'action suivante...")

for agent in world.agents:
    print(agent.num)
    print(agent.has_arrived)
    print()
    print(agent.is_alive)
