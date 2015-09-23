#!/usr/bin/python

import maze_lib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})

def drawMaze(maze_file = "eazy_maze.txt"):
lines = [
[(293, 7), (289, 130)],
[(289, 130), (6, 134)],
[(6, 134), (8, 5)],
[(8, 5), (292, 7)],
[(241, 130), (58, 65)],
[(114, 7), (73, 42)],
[(130, 91), (107, 46)],
[(196, 8), (139, 51)],
[(219, 122), (182, 63)],
[(267, 9), (214, 63)],
[(271, 129), (237, 88)]
]


lc = mc.LineCollection(lines, linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.scatter(270, 100)
ax.scatter(197.786, 123.179, c="red")
ax.scatter(30, 22, c="green")
ax.autoscale()
ax.margins(0.1)
plt.show()

if __name__ == "__main__":
  testMulti()