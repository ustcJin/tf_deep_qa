# -*- coding=utf-8 -*-
import cPickle
import numpy
import sys
import os

##########Batcher############
# datasets: list, support multi-data
# batch_size: all known
class Batcher:
	def __init__(self, datasets, batch_size = 100):
		self.datasets = datasets
		self.batch_size = batch_size
		self.samples = datasets[0].shape[0]
		self.n_batches = self.samples / batch_size
		print self.n_batches
	def __iter__(self):
		for i in xrange(self.n_batches):
			yield [x[i * self.batch_size : (i + 1) * self.batch_size] for x in self.datasets]



## test
if __name__ == '__main__':
	#test_data_1 = numpy.random.randn(100, 10)
	#test_data_2 = numpy.random.randn(100, 10)
	test_data_1 = numpy.ones([100,10])
	test_data_2 = numpy.zeros([100,10])
	batcher = Batcher([test_data_1, test_data_2], batch_size = 10)
	for i in batcher:
		print i
		print '########'
