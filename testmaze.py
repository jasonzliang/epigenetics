#!/usr/bin/python

import glob
import maze_lib
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import collections as mc
np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})

def drawMaze(maze_file = "easy_maze4.txt", endpts = None, paths = None, keyPoints = None,
  outfile = None, markerStyle = None, lineColor = None, lineLabels = None,
  show = False):
  if outfile == None:
    outfile = maze_file
  f = open(maze_file)
  data = f.readlines()
  num_lines = float(data[0].rstrip())
  startposition = [float(x) for x in data[1].rstrip().split()]
  goal = [float(x) for x in data[3].rstrip().split()]
  lines = []
  for line in data[4:]:
    __line = [float(x) for x in line.rstrip().split()]
    lines.append([[__line[0], __line[1]], [__line[2], __line[3]]])
  assert(num_lines == len(lines))

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
      if markerStyle and lineColor and lineLabels:
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=1,
        alpha=0.3, marker=markerStyle[i], color=lineColor[i], label=lineLabels[i])
      else:
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=1,
        alpha=0.2, marker='.', color=np.random.rand(3,1))
  if keyPoints:
    plt.scatter([x[0] for x in keyPoints], [x[1] for x in keyPoints], c="black", 
      label="final pos", s=30, alpha=0.5)
  plt.legend(loc="upper left", bbox_to_anchor=(1,1))
  # ax.autoscale()
  # ax.margins(0.1)
  if show:
    plt.show()
  plt.savefig(outfile + ".png", bbox_inches="tight", dpi=200)

def drawMazeRecordFromFile(maze_file = "easy_maze_bn2.txt",
  maze_record_dir = "neat_records"):
  for maze_record in glob.glob(maze_record_dir + "/*.txt"):
    print "drawing " + maze_record
    outfile = maze_record[:-4] + ".png"
    f = open(maze_record)
    path = []
    for line in f.readlines():
      data = line.rstrip().split()
      path.append((data[1], data[2]))
    drawMaze(maze_file = maze_file, endpts = None, paths = [path], 
      outfile = outfile)

if __name__ == "__main__":
  if len(sys.argv) == 2:
    drawMaze(sys.argv[1])
  else:
    drawMaze()
