#!/usr/bin/python

import maze_lib
import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'all':lambda x: str(x) + ','})

genes = np.random.random(130)
print genes.dtype
ac = np.arange(10, dtype=np.float64)

maze_lib.SetUp()
maze_lib.SetACFlag(False)

for i in xrange(10):
	print maze_lib.EvalNetwork(genes, ac)
	print maze_lib.ReturnActivityCounter()

print "set AC flag to true"

maze_lib.SetACFlag(True)

for i in xrange(10):
	print maze_lib.EvalNetwork(genes, ac)
	print maze_lib.ReturnActivityCounter()
