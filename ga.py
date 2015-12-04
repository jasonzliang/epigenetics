#!/usr/bin/python

import numpy as np
import time
import random, math, sys, os
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import testmaze
import maze_lib

np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})

class genetic(object):
  def __init__(self, genomeSize, hiddenSize, maze_file, popMax=1000, 
               parameters=[0.1, 0.5, 0.5, 0.5, 2.0, 0.15],
               maxEvals=22000, useScaling=False, draw=False, 
               useBackprop=False, **keywords):
    print "maze: ", maze_file
    print "genome size: ", genomeSize
    print "hidden size: ", hiddenSize
    self.maze_file = maze_file
    self.useScaling = useScaling
    self.popMax = popMax
    self.genomeSize = genomeSize
    self.hiddenSize = hiddenSize
    self.maxEvals = maxEvals
    self.draw = draw
    if useBackprop:
      print "USING BACKPROP!!!!!"
      self.useBackprop, self.bpProb, self.epochs, self.numSamples, self.randomShuffle \
        = useBackprop
    else:
      self.useBackprop = useBackprop

    self.setParameters(parameters)    
    self.reset()
    
  def setParameters(self, parameters):
    self.mutRate, self.mutAmount, self.crossRate, self.replacementRate, \
      self.initRange, self.popSize = parameters
    self.popSize = max(2, int(self.popMax*self.popSize))    
    self.replacementRate = int(math.ceil(self.popSize*self.replacementRate))
    if self.replacementRate > 0 and self.replacementRate % 2 != 0:
      self.replacementRate -= 1
      
  def reset(self):
    self.numGen = self.numEval = 0
    self.fitness = np.zeros(self.popSize) + 1e-12
    self.population = np.random.uniform(-self.initRange, self.initRange, 
      (self.popSize, self.genomeSize))

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
    genome_copy = np.copy(genome)
    fitness, x, y = maze_lib.EvalNetwork(genome)
    # print genome
    # sys.exit()
    # newGenome = maze_lib.ReturnWeights()
    # newGenome = newGenome.astype(np.float64)
    return fitness

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
  
  def drawBeforeAfterTeach(self, master, student, learned_student, bpPositions):
    if self.draw and random.random() < 1.0/(2.0*float(self.popSize)):
      paths = []
      endpts = []
      maze_lib.SetVerbosity(True)
      dataMaster = maze_lib.EvalNetwork(master)
      dataBefore = maze_lib.EvalNetwork(student)
      dataAfter = maze_lib.EvalNetwork(learned_student)
      maze_lib.SetVerbosity(False)

      paths.append(dataMaster[3:])
      paths.append(dataBefore[3:])
      paths.append(dataAfter[3:])
      endpts.append((dataMaster[1], dataMaster[2]))
      endpts.append((dataBefore[1], dataBefore[2]))
      endpts.append((dataAfter[1], dataAfter[2]))

      print "drawing before and after backprop learning"
      testmaze.drawMaze(self.maze_file, endpts, paths, bpPositions,
        self.maze_file + "_" + str(self.numEval) + "_comp", 
        markerStyle=['.', 'x', '+'], lineColor=['b', 'r', 'g'], 
        lineLabels=['teacher', 'before', 'after'])
      # update = learned_student - student
      # plt.clf()
      # plt.hist(update, 20, color='green', alpha=0.8)
      # plt.title("Histogram of Weight Updates")
      # plt.savefig(str(self.numEval) + "_hist.png", bbox_inches="tight")

  def teach(self, master, student):
    learned_student, bpPositions = maze_lib.Backprop(master, student, 
      self.epochs, self.numSamples, self.randomShuffle)
    learned_student = learned_student.astype(np.float64)
    self.drawBeforeAfterTeach(master, student, learned_student, bpPositions)
    # print bpPositions
    # sys.exit()
    return learned_student

  def teach2(self):
    frac=self.bpProb/2
    for i in xrange(int(frac * self.popSize)):
      indices = random.sample(xrange(self.popSize), self.popSize/10)
      master_index = indices[np.argmax(self.fitness[indices])]
      student_index = indices[np.argmin(self.fitness[indices])]
      student = self.population[student_index,:]
      master = self.population[master_index,:]
      learned_student, bpPositions = \
        maze_lib.Backprop(master, student, self.epochs, self.numSamples, self.randomShuffle)
      learned_student = learned_student.astype(np.float64)
      self.drawBeforeAfterTeach(master, student, learned_student, bpPositions)
      self.population[student_index,:] = learned_student

  def step(self):      
    indices = np.argsort(self.fitness)
    self.calculateStats()
    newPopulation = self.population[indices,:]
    newFitness = self.fitness[indices]
      
    for i in xrange(0,self.replacementRate,2):
      a_i = self.fitnessSelection()
      b_i = self.fitnessSelection()
      p_a = self.population[a_i,:]
      p_b = self.population[b_i,:]
      a = np.copy(p_a); b = np.copy(p_b)
      a,b = self.crossOver(a,b)
      a = self.mutate(a)
      b = self.mutate(b)
      if self.useBackprop and random.random() < self.bpProb:
        # teaching method using best in population
        # bestGenome = self.population[np.argmax(self.fitness),:]
        # a = self.teach(bestGenome, a)
        # b = self.teach(bestGenome, b)

        # teaching method using parent
        a = self.teach(p_a, a)
        b = self.teach(p_b, b)
      newPopulation[i,:] = a
      newPopulation[i+1,:] = b
    
    for i in xrange(self.replacementRate):
      newFitness[i] = self.getFitness(newPopulation[i,:])

    self.population = newPopulation
    self.fitness = newFitness
    # if self.useBackprop:
    #   self.teach2()
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

def testGA(nTrials=1, nGen=250, useBackprop=(True, 0.5, 2, 25, False),
  maze_file = "medium_maze.txt", draw=True, learningRate=0.05, timesteps=400):

  maze_lib.SetUp(learningRate, timesteps)
  maze_lib.SetMazePath(os.getcwd() + "/" + maze_file)
  maze_lib.SetVerbosity(False)

  if not useBackprop or not useBackprop[0]:
    useBackprop = False

  gen = None
  avgBestFit = []
  avgMeanFit = []
  endpts = []
  paths = []
  for k in xrange(nTrials):
    print "trial number:", k+1
    # reload(maze_lib)
    # maze_lib.SetUp(learningRate)
    # maze_lib.SetMazePath(os.getcwd() + "/" + maze_file)
    # maze_lib.SetVerbosity(False)
    data = []
    ga = genetic(genomeSize=maze_lib.GetNetworkWeightCount(),
      hiddenSize=maze_lib.GetNetworkHiddenUnitCount(),
      maxEvals=1e308, draw=draw, useBackprop=useBackprop,
      maze_file=maze_file) 
    for i in xrange(nGen):
      ga.step()
      if ga.numGen % 5 == 0:
        data.append(ga.printStats(printGenome=False))

    maze_lib.SetVerbosity(True)
    print ga.bestGenome
    results = maze_lib.EvalNetwork(ga.bestGenome)
    maze_lib.SetVerbosity(False)
    fitness = results[0]
    paths.append(results[3:])
    endpts.append((results[1], results[2]))

    if not gen:
      gen = [x[0] for x in data] 
    avgBestFit.append([x[2] for x in data])
    avgMeanFit.append([x[3] for x in data])


  if draw:
    return

  avgBestFit = np.mean(np.vstack(avgBestFit), axis=0)
  avgMeanFit = np.mean(np.vstack(avgMeanFit), axis=0)
  plt.figure(figsize=(14,10))
  plt.clf()
  # plt.ylim(ymin=0)
  plt.plot(gen, avgBestFit, label="best fit")
  plt.plot(gen, avgMeanFit, label="avg fit")
  plt.xlabel("generation")
  plt.ylabel("fitness")
  plt.legend(loc='lower right')
  plt.grid()

  out_str = str(time.time()) + "_" + maze_file + "_bp" + \
    str(useBackprop) + "_t" + str(timesteps)
  plt.savefig(out_str + "_graph.png", bbox_inches="tight", dpi=200)

  testmaze.drawMaze(maze_file, endpts, paths, outfile=out_str + "_vis")
  testmaze.drawMaze(maze_file, endpts, None, outfile=out_str + "_vis2")
  
if __name__ == "__main__":
  testGA()
    