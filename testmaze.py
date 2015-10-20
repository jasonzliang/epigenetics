#!/usr/bin/python

import maze_lib
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import collections as mc
np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})

def drawMaze(maze_file = "easy_maze4.txt", endpts = None, paths = None, 
  outfile = None, markerStyle = None, lineColor = None):
  if outfile == None:
    outfile = maze_file
  f = open(maze_file)
  data = f.readlines()
  startposition = [float(x) for x in data[1].rstrip().split()]
  goal = [float(x) for x in data[3].rstrip().split()]
  lines = []
  for line in data[4:]:
    __line = [float(x) for x in line.rstrip().split()]
    lines.append([[__line[0], __line[1]], [__line[2], __line[3]]])

  lc = mc.LineCollection(lines, linewidths=2)
  fig, ax = plt.subplots()
  fig.set_figwidth(14)
  fig.set_figheight(10)
  ax.add_collection(lc)
  plt.scatter(startposition[0], startposition[1], c="blue", label="start", s=40)
  plt.scatter(goal[0], goal[1], c="green", label="goal", s=40)
  if endpts:
    plt.scatter([x[0] for x in endpts], [x[1] for x in endpts], c="red", 
      label="final pos", s=40, alpha=0.5)
  if paths:
    for i, path in enumerate(paths):
      if markerStyle and lineColor:
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=1,
        alpha=0.5, marker=markerStyle[i], color=lineColor[i])
      else:
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=1,
        alpha=0.25, marker='.', color="r")

  # ax.autoscale()
  # ax.margins(0.1)
  plt.legend(loc="upper left", bbox_to_anchor=(1,1))
  plt.savefig(outfile + ".png", bbox_inches="tight", dpi=200)
  # plt.show()

if __name__ == "__main__":
  if len(sys.argv) == 2:
    drawMaze(sys.argv[1])
  else:
    drawMaze()
