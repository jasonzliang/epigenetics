#!/usr/bin/python

import numpy as np
import time
import random, math, sys, os
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import testmaze
import maze_lib

np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})
maze_lib.SetUp(False, False)

class genetic(object):
  def __init__(self, genomeSize, hiddenSize, popMax=1000, 
               parameters=[0.2, 0.3, 0.5, 0.5, 1.0, 0.15],
               maxEvals=22000, useScaling=False, draw=False, **keywords):
    print "genome size: ", genomeSize
    print "hidden size: ", hiddenSize
    self.useScaling = useScaling
    self.popMax = popMax
    self.genomeSize = genomeSize
    self.hiddenSize = hiddenSize
    self.maxEvals = maxEvals
    self.draw = draw

    self.setParameters(parameters)    
    self.reset()
    
  def setParameters(self, parameters):
    self.mutRate, self.mutAmount, self.crossRate, self.replacementRate, self.initRange, self.popSize = parameters
    self.popSize = max(2, int(self.popMax*self.popSize))    
    self.replacementRate = int(math.ceil(self.popSize*self.replacementRate))
    if self.replacementRate > 0 and self.replacementRate % 2 != 0:
      self.replacementRate -= 1
      
  def reset(self):
    self.numGen = self.numEval = 0
    self.fitness = np.zeros(self.popSize) + 1e-12
    self.population = np.random.uniform(-self.initRange, self.initRange, (self.popSize, self.genomeSize))

  def isTerminal(self):
    return self.numEval > self.maxEvals
    
  def tournamentSelection(self, tSize=2):
    indices = random.sample(xrange(self.popSize), tSize)
    return indices[np.argmax(self.fitness[indices])]
  
  def fitnessSelection(self):
    if self.useScaling or self.currMinFitness <= 0.0:
      bestFitness = self.scaledBestFitness
      fitnesses = self.scaledFitness
    else:
      bestFitness = self.currBestFitness
      fitnesses = self.fitness
    while True:
      i = random.randrange(self.popSize)
      if random.random() < fitnesses[i]/bestFitness:
        return i

  def crossOver(self, ga, b):
    r = np.random.random(self.genomeSize) < self.crossRate
    ga[r], b[r] = b[r], ga[r]
    return ga, b

  def mutate(self, a):
    x = np.random.normal(scale=self.mutAmount, size=self.genomeSize)
    r = np.random.random(self.genomeSize) < self.mutRate
    a[r] += x[r]
    return a
  
  def getFitness(self, genome):
    self.numEval += 1
    if self.draw and random.random() < 0.003:
      paths = []
      endpts = []
      maze_lib.SetVerbosity(True)
      dataBefore = maze_lib.EvalNetwork(genome,
        np.percentile(self.fitness, 75))
      newGenome = maze_lib.ReturnWeights()
      newGenome = newGenome.astype(np.float64)
      dataAfter = maze_lib.EvalNetwork(newGenome,
        np.percentile(self.fitness, 75))
      paths.append(dataBefore[3:])
      endpts.append((dataBefore[1], dataBefore[2]))
      paths.append(dataAfter[3:])
      endpts.append((dataAfter[1], dataAfter[2]))
      maze_lib.SetVerbosity(False)

      print "drawing before and after hebbian learning"
      testmaze.drawMaze("easy_maze4.txt", endpts, paths, 
        str(self.numEval) + "_comp", 
        markerStyle = ['.', 'x'], lineColor = ['r', 'g'])
      update = newGenome - genome
      plt.clf()
      plt.hist(update, 20, color='green', alpha=0.8)
      plt.title("Histogram of Weight Updates")
      plt.savefig(str(self.numEval) + "_hist.png", bbox_inches="tight")

      return dataBefore[0], newGenome

    fitness, x, y = maze_lib.EvalNetwork(genome,
      np.percentile(self.fitness, 75))
    newGenome = maze_lib.ReturnWeights()
    newGenome = newGenome.astype(np.float64)
    return fitness, newGenome


  def getBestFitness(self):
    return np.max(self.fitness)
  
  def getMeanFitness(self):
    return np.mean(self.fitness)
  
  def getQuartileFitness(self, p=75):
    return np.percentile(self.fitness, q=p)
  
  def getMedianFitness(self):
    return np.median(self.fitness)
  
  def getBestGenome(self):
    return self.population[np.argmax(self.fitness),:]
  
  def linearRankScaling(self, indices):
    self.scaledFitness = indices + 1.0
    
  def positiveScaling(self, indices):
    self.scaledFitness = (self.fitness - np.min(self.fitness)) + 1e-12
    
  def calculateStats(self):
    self.currBestFitness = np.max(self.fitness)
    self.currMeanFitness = np.mean(self.fitness)
    self.currMinFitness = np.min(self.fitness)
    
    if self.useScaling:
      self.scaledBestFitness = np.max(self.scaledFitness)
      self.scaledMeanFitness = np.mean(self.scaledFitness)
      self.scaledMinFitness = np.min(self.scaledFitness)
    
  def step(self):      
    indices = np.argsort(self.fitness)
    self.calculateStats()
    newPopulation = self.population[indices,:]
      
    for i in xrange(0,self.replacementRate,2):
      a_i = self.fitnessSelection()
      b_i = self.fitnessSelection()
      a = self.population[a_i,:]
      b = self.population[b_i,:]
      a = np.copy(a); b = np.copy(b)
      a,b = self.crossOver(a,b)
      a = self.mutate(a)
      b = self.mutate(b)
      
      newPopulation[i,:] = a
      newPopulation[i+1,:] = b
    
    for i in xrange(self.replacementRate):
      self.fitness[i], newPopulation[i,:] = self.getFitness(newPopulation[i,:])

    self.population = newPopulation
    self.numGen += 1

  def printStats(self, printGenome=False, printAC=True):
    bestFitness = np.max(self.fitness)
    meanFitness = np.mean(self.fitness)
    minFitness = np.min(self.fitness)
    print "gen: " + str(self.numGen) + " eval: " + str(self.numEval) + \
      " bestFitness: " + str(bestFitness) + " avgFitness: " + str(meanFitness) + \
      " minFitness: " + str(minFitness)
    self.bestGenome = self.population[np.argmax(self.fitness),:]
    if printGenome:
      print "best genome: " + str(self.bestGenome)
    return self.numGen, self.numEval, bestFitness, meanFitness, minFitness

def testGA(nTrials=30, nGen=200, useAC=True, recurrent=False, 
  maze_file = "easy_maze4.txt", draw=False):

  maze_lib.SetUp(useAC, recurrent)
  maze_lib.SetMazePath(os.getcwd() + "/" + maze_file)
  maze_lib.SetVerbosity(False)

  gen = None
  avgBestFit = []
  avgMeanFit = []
  endpts = []
  paths = []
  for k in xrange(nTrials):
    data = []
    ga = genetic(genomeSize=maze_lib.GetNetworkWeightCount(),
      hiddenSize=maze_lib.GetNetworkHiddenUnitCount(),
      maxEvals=1e308, draw=draw) 
    for i in xrange(nGen):
      ga.step()
      if ga.numGen % 5 == 0:
        data.append(ga.printStats(printGenome=False))
    # ga.printStats(printGenome=True)

    maze_lib.SetVerbosity(True)
    print ga.bestGenome
    results = maze_lib.EvalNetwork(ga.bestGenome, 0)
    maze_lib.SetVerbosity(False)
    fitness = results[0]
    paths.append(results[3:])
    endpts.append((results[1], results[2]))

    if not gen:
      gen = [x[0] for x in data] 
    avgBestFit.append([x[2] for x in data])
    avgMeanFit.append([x[3] for x in data])

  avgBestFit = np.mean(np.vstack(avgBestFit), axis=0)
  avgMeanFit = np.mean(np.vstack(avgMeanFit), axis=0)
  plt.figure(figsize=(14,10))
  plt.clf()
  plt.ylim((0, 300))
  plt.plot(gen, avgBestFit, label="best fit")
  plt.plot(gen, avgMeanFit, label="avg fit")
  plt.xlabel("generation")
  plt.ylabel("fitness")
  plt.legend(loc='lower right')
  plt.grid()
  plt.savefig(str(time.time()) + "_" + maze_file + "_ac" + str(useAC) + 
    "_recur" + str(recurrent) + "_graph.png", bbox_inches="tight")

  testmaze.drawMaze(maze_file, endpts, paths, str(time.time()) + "_" + maze_file + 
    "_ac" + str(useAC) + "_recur" + str(recurrent) + "_vis")
  
if __name__ == "__main__":
  testGA()
    