import cv2
from lle import World, Action


def show(world: World):
    img = world.get_image()
    cv2.imshow("Visualisation", img)
    cv2.waitKey(1)


world = World.level(1)
world.reset()
show(world)
path = [Action.SOUTH] * 5
path += [Action.EAST] * 3
path += [Action.SOUTH] * 5
path += [Action.WEST] * 3
for action in path:
    events = world.step(action)
    print(events)
    show(world)
    input("Appuyez sur 'enter' pour passer à l'action suivante...")

for agent in world.agents:
    print(agent.has_arrived)
    print(agent.is_alive)
