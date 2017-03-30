#! /bin/python
# -*- coding=utf8 -*-

import tensorflow as tf
import numpy
import os
import sys
import time
from getBatch import Batcher

class Layer(object):
	def __init__(self):
		self.params = []
		self.weights = []
		self.biases = []
	def output_func(self, input):
		raise NotImplementedError("This is virtual class, need inherited.")
	def set_input(self, input):
		self.output = self.output_func(input)
	def __repr__(self):
		return '{}'.format(self.__class__.__name__)

class FeedForwardNet(Layer):
	def __init__(self, layers = None, name = None):
		super(FeedForwardNet, self).__init__()	
		self.layers = layers
		self.name = name
		for layer in self.layers:
			self.weights.extend(layer.weights)
			self.biases.extend(layer.biases)
			self.params.extend(layer.weights + layer.biases)
	def output_func(self, input):
		cur_input = input
		for layer in self.layers:
			layer.set_input(cur_input)
			cur_input = layer.output
		return cur_input 
	def __repr__(self):
		layers_str = '\n'.join(['\t{}'.format(line) for layer in self.layers for line in str(layer).splitlines()])
		return '{}'.format(layers_str)

class ParallelLookupTable(FeedForwardNet):
	def output_func(self, input):
		layers_out = []
		assert len(input) == len(self.layers)
		for x, layer in zip(input, self.layers):
			layer.set_input(x)
			layers_out.append(layer.output)
		return tf.concat(layers_out, 2)

class LookupTable(Layer):
	def __init__(self, W=None, batch_size=100, filter_width=5, width = 28, is_var = True):
		super(LookupTable, self).__init__()
		self.W = W
		self.filter_width = filter_width
		self.batch_size = batch_size
		self.width = width
		self.pad_matrix = numpy.zeros([self.batch_size, self.filter_width - 1, self.width])
		if is_var:
			self.weights = [self.W]
	def output_func(self, input):
		out = tf.concat([self.pad_matrix, tf.to_float(tf.nn.embedding_lookup(self.W, input)), self.pad_matrix], 1)
		return out

class Conv2d(Layer):
	def __init__(self, filter_shape, filter_width = 5, batch_size = 100, height = 28, width = 28, nkernals = 1):
		super(Conv2d, self).__init__()
		bound = numpy.sqrt(1. / numpy.prod(filter_shape[0:2]))
		self.filter = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-bound, maxval=bound))
		self.filter_width = filter_width
		self.batch_size = batch_size
		self.height = height
		self.width = width
		self.nkernals = nkernals
		self.weights = [self.filter]
	def output_func(self, input):
		input = tf.reshape(input, [-1, self.height, self.width, self.nkernals])
		return tf.nn.conv2d(input, self.filter, strides = [1,1,1,1], padding = 'VALID')

class Activation(Layer):
	def __init__(self, activation = tf.tanh, nkernals = 100):
		super(Activation, self).__init__()
		self.activation = activation
		self.b = tf.Variable(tf.zeros(shape=[nkernals]))
		self.nkernals = nkernals
		self.biases = [self.b]
	def output_func(self, input):
		return self.activation(input +  tf.reshape(self.b, [1,1,1,self.nkernals]))

class Maxpool(Layer):
	def __init__(self, ksize = [1,1,1,1], strides = [1,1,1,1], padding = 'VALID'):
		super(Maxpool, self).__init__()
		self.ksize = ksize
		self.strides = strides
		self.padding = padding
	def output_func(self, input):
		return tf.nn.max_pool(input, self.ksize, self.strides, self.padding)

class Flatten(Layer):
	def __init__(self, batch_size=100):
		super(Flatten, self).__init__()
		self.batch_size = batch_size
	def output_func(self, input):
		return tf.reshape(input, shape=[self.batch_size, 1, -1])

class PairCombine(Layer):
	def __init__(self, shape1, shape2):
		super(PairCombine, self).__init__()
		self.shape1 = shape1
		self.shape2 = shape2
		self.W = tf.Variable(tf.zeros(shape=[shape1, shape2]))
		self.weights = [self.W]
	def output_func(self, input):
		q, a= input[0], input[1]
		mid1 = tf.reshape(a, shape=[-1, self.shape1])
		mid2 = tf.matmul(mid1, self.W)
		mid3 = tf.reshape(mid2, shape=[-1, 1, self.shape2])
		sim = tf.matmul(q, mid3, transpose_b=True)
		return tf.concat([q, sim, a], 2);

class Linear(Layer):
	def __init__(self, n_in, n_out, activation=tf.tanh):
		super(Linear, self).__init__()
		bound=numpy.sqrt(1.2/(n_in+n_out))
		self.W = tf.Variable(tf.random_uniform(shape=[n_in, n_out], minval=-bound, maxval=bound))
		self.b = tf.Variable(tf.zeros(shape=[n_out]))
		self.activation = tf.tanh
		self.n_in = n_in
		self.n_out = n_out
		self.weights = [self.W]
		self.biases = [self.b]
	def output_func(self, input):
		mid1 = tf.reshape(input, [-1, self.n_in])
		mid2 = tf.add(tf.matmul(mid1, self.W), self.b)
		mid3 = tf.reshape(mid2, shape=[-1, self.n_out])
		return self.activation(mid3)

class LR(Layer):
	def __init__(self, n_in, n_out):
		super(LR, self).__init__()
		self.W = tf.Variable(tf.zeros(shape=[n_in, n_out]))
		self.b = tf.Variable(tf.zeros(shape=[n_out]))
		self.n_in = n_in
		self.n_out = n_out
		self.weights = [self.W]
		self.biases = [self.b]
	def output_func(self, input):
		mid1 = tf.reshape(input, [-1, self.n_in])
		mid2 = tf.add(tf.matmul(mid1, self.W), self.b)
		mid3 = tf.reshape(mid2, shape=[-1, self.n_out])
		return mid3
