#!/usr/bin/python

import numpy as np
import time
import random, math, sys, os
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import maze_lib

np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})
maze_lib.SetUp(False, False)

class genetic(object):
  def __init__(self, genomeSize, hiddenSize, popMax=1000, 
               parameters=[0.2, 0.3, 0.5, 0.5, 1.0, 0.15],
               maxEvals=22000, useScaling=False, **keywords):
    print "genome size: ", genomeSize
    print "hidden size: ", hiddenSize
    self.useScaling = useScaling
    self.popMax = popMax
    self.genomeSize = genomeSize
    self.hiddenSize = hiddenSize
    self.maxEvals = maxEvals

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
    self.activityCounter = np.ones((self.popSize, self.hiddenSize)) * 0.00001

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
  
  def getFitness(self, genome, ac):
    self.numEval += 1
    fitness = maze_lib.EvalNetwork(genome, ac)
    ac = maze_lib.ReturnActivityCounter()
    return fitness, ac

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
    newActivityCounter = self.activityCounter[indices,:]
      
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

      a_AC = self.activityCounter[a_i,:]
      b_AC = self.activityCounter[b_i,:]
      newActivityCounter[i,:] = (a_AC + b_AC) * 0.5
      newActivityCounter[i+i,:] = (b_AC + a_AC) * 0.5
    
    for i in xrange(self.replacementRate):
      self.fitness[i], newActivityCounter[i,:] = self.getFitness(newPopulation[i,:], newActivityCounter[i,:])
           
    self.population = newPopulation
    self.activityCounter = newActivityCounter
    self.numGen += 1

  def printStats(self, printGenome=False, printAC=True):
    bestFitness = np.max(self.fitness)
    meanFitness = np.mean(self.fitness)
    minFitness = np.min(self.fitness)
    print "gen: " + str(self.numGen) + " eval: " + str(self.numEval) + " bestFitness: " + str(bestFitness) + " avgFitness: " + str(meanFitness) + " minFitness: " + str(minFitness)
    if printGenome:
      bestGenome = self.population[np.argmax(self.fitness),:]
      print "best genome: " + str(bestGenome)
    if printAC:
      bestAC = self.activityCounter[np.argmax(self.fitness),:]
      print "best activity counter: " + str(bestAC)
    return self.numGen, self.numEval, bestFitness, meanFitness, minFitness

def testMulti():
  testGA(useAC=False, recurrent=False)
  testGA(useAC=False, recurrent=True)
  testGA(useAC=True, recurrent=False)
  testGA(useAC=True, recurrent=True)

def testGA(nGen=500, outputName="results", useAC=False, recurrent=True):
  maze_lib.SetUp(useAC, recurrent)
  data = []
  ga = genetic(genomeSize=maze_lib.GetNetworkWeightCount(),
    hiddenSize=maze_lib.GetNetworkHiddenUnitCount(),
    maxEvals=1e308) 
  for i in xrange(nGen):
    ga.step()
    if ga.numGen % 10 == 0:
      data.append(ga.printStats(printGenome=False))
  ga.printStats(printGenome=True)

  gen = [x[0] for x in data]
  bestFit = [x[2] for x in data]
  meanFit = [x[3] for x in data]

  plt.clf()
  plt.plot(gen, bestFit, label="best fit")
  plt.plot(gen, meanFit, label="avg fit")
  plt.xlabel("generation")
  plt.ylabel("fitness")
  plt.legend(loc='upper left')
  plt.grid()
  plt.savefig(outputName + "_ac" + str(useAC) + "_recur" + str(recurrent) + ".png")
  
if __name__ == "__main__":
  testMulti()
    