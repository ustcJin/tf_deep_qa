#! /bin/python
# -*- coding=utf8 -*-

import tensorflow as tf
import numpy
import os
import sys
import time
from getBatch import Batcher

class Layer(object):
	def __init__(self):pass
	def output_func(self, input):
		raise NotImplementedError("This is virtual class, need inherited.")
	def set_input(self, input):
		self.output = self.output_func(input)
	def __repr__(self):
		return '{}'.format(self.__class__.__name__)

class FeedForwardNet(Layer):
	def __init__(self, layers = None):
		self.layers = layers
	def output_func(self, input):
		cur_input = input
		for layer in self.layers:
			layer.set_input(cur_input)
			cur_input = layer.output
		return cur_input 

class ParallelLookupTable(FeedForwardNet):
	def output_func(self, input):
		layers_out = []
		assert len(input) == len(self.layers)
		for x, layer in zip(input, self.layers):
			layer.set_input(x)
			layers_out.append(layer.output)
		return tf.concat(layers_out, 2)

class LookupTable(Layer):
	def __init__(self, W=None):
		super(LookupTable, self).__init__()
		self.W = W
	def output_func(self, input):
		return tf.to_float(tf.nn.embedding_lookup(self.W, input))

class Conv2d(Layer):
	def __init__(self, filter_shape, filter_width = 5, batch_size = 100, height = 28, width = 28, nkernals = 1):
		self.filter = tf.Variable(tf.random_normal(shape=filter_shape))
		self.filter_width = filter_width
		self.batch_size = batch_size
		self.height = height
		self.width = width
		self.nkernals = nkernals
	def output_func(self, input):
		input = tf.reshape(input, [-1, self.height, self.width, self.nkernals])
		padding = tf.random_normal([self.batch_size, self.filter_width - 1, self.width, self.nkernals])
		input = tf.concat([padding, input, padding], 1)
		return tf.nn.conv2d(input, self.filter, strides = [1,1,1,1], padding = 'VALID')

class Activation(Layer):
	def __init__(self, actvivation = tf.tanh):
		self.actvivation = actvivation
	def output_func(self, input):
		return self.actvivation(input)

class Maxpool(Layer):
	def __init__(self, ksize = [1,1,1,1], strides = [1,1,1,1], padding = 'VALID'):
		self.ksize = ksize
		self.strides = strides
		self.padding = padding
	def output_func(self, input):
		return tf.nn.max_pool(input, self.ksize, self.strides, self.padding)

class Flatten(Layer):
	def __init__(self, batch_size=100):
		self.batch_size = batch_size
	def output_func(self, input):
	#	return tf.contrib.layers.flatten(input)
		return tf.reshape(input, shape=[self.batch_size, 1, -1])

class PairCombine(Layer):
	def __init__(self, shape1, shape2):
		self.shape1 = shape1
		self.shape2 = shape2
		self.W = tf.Variable(tf.random_normal(shape=[shape1, shape2]))
	def output_func(self, input):
		q, a= input[0], input[1]
		mid1 = tf.reshape(a, shape=[-1, self.shape1])
		mid2 = tf.matmul(mid1, self.W)
		mid3 = tf.reshape(mid2, shape=[-1, 1, self.shape2])
		sim = tf.matmul(q, mid3, transpose_b=True)
		return tf.concat([q, sim, a], 2);

class Linear(Layer):
	def __init__(self, n_in, n_out, activation=tf.tanh):
		self.W = tf.Variable(tf.random_normal(shape=[n_in, n_out]))
		self.b = tf.Variable(tf.random_normal(shape=[n_out]))
		self.activation = tf.tanh
		self.n_in = n_in
		self.n_out = n_out
	def output_func(self, input):
		mid1 = tf.reshape(input, [-1, self.n_in])
		mid2 = tf.add(tf.matmul(mid1, self.W), self.b)
		mid3 = tf.reshape(mid2, shape=[-1, self.n_out])
		return self.activation(mid3)

class LR(Layer):
	def __init__(self, n_in, n_out):
		self.W = tf.Variable(tf.random_normal(shape=[n_in, n_out]))
		self.b = tf.Variable(tf.random_normal(shape=[n_out]))
		self.n_in = n_in
		self.n_out = n_out
	def output_func(self, input):
		mid1 = tf.reshape(input, [-1, self.n_in])
		mid2 = tf.add(tf.matmul(mid1, self.W), self.b)
		mid3 = tf.reshape(mid2, shape=[-1, self.n_out])
		return mid3

if __name__ == '__main__':
	#LOAD
	data_dir = '../TRAIN'

	q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
	a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
	q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
	a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
	y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))
	vocab_emb = numpy.load(os.path.join(data_dir, 'emb_vectors.bin.npy'))

	q_dev = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
	a_dev = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
	q_overlap_dev = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
	a_overlap_dev = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
	y_dev = numpy.load(os.path.join(data_dir, 'test.labels.npy'))

	#Load PRINT
	print 'y_train', numpy.unique(y_train, return_counts=True)
	print 'word embeddings ndim', vocab_emb.shape[1]

	#Init
	q_max_sent_size = q_train.shape[1]
	a_max_sent_size = a_train.shape[1]
	word_dim = vocab_emb.shape[1]
	filter_width = 5
	kernals = 10
	batch_size = 10
	overlap_ndim = 5
	n_class = 2
	learning_rate = 0.001
	overlap_word_max_id = numpy.max(q_overlap_train)
	filter_shape = [filter_width, word_dim + overlap_ndim, 1, kernals]
	vocab_emb_overlap = tf.Variable(tf.random_normal(shape=[overlap_word_max_id+1, overlap_ndim]))
	x_q = tf.placeholder(tf.int32, [None, q_max_sent_size])
	x_q_overlap = tf.placeholder(tf.int32, [None, q_max_sent_size])
	x_a = tf.placeholder(tf.int32, [None, a_max_sent_size])
	x_a_overlap = tf.placeholder(tf.int32, [None, a_max_sent_size])
	y = tf.placeholder(tf.int32, [None, n_class])

	#Define Net
	lookup_words = LookupTable(W=vocab_emb)
	lookup_words_overlap = LookupTable(W=vocab_emb_overlap)
	lookup_words = ParallelLookupTable([lookup_words, lookup_words_overlap])
	conv = Conv2d(filter_shape=filter_shape, filter_width=filter_width, batch_size=batch_size, height=q_max_sent_size, width=overlap_ndim+word_dim)
	non_line = Activation(tf.tanh)
	max_pool = Maxpool(ksize=[1, q_max_sent_size+filter_width-1, 1, 1])
	flatten = Flatten(batch_size=batch_size)
	net_q = FeedForwardNet([lookup_words, conv, non_line, max_pool, flatten])
	net_q.set_input([x_q, x_q_overlap])

	lookup_words = LookupTable(W=vocab_emb)
	lookup_words_overlap = LookupTable(W=vocab_emb_overlap)
	lookup_words = ParallelLookupTable([lookup_words, lookup_words_overlap])
	conv = Conv2d(filter_shape=filter_shape, filter_width=filter_width, batch_size=batch_size, height=a_max_sent_size, width=overlap_ndim+word_dim)
	non_line = Activation(tf.tanh)
	max_pool = Maxpool(ksize=[1, a_max_sent_size+filter_width-1, 1, 1])
	flatten = Flatten(batch_size=batch_size)
	net_a = FeedForwardNet([lookup_words, conv, non_line, max_pool, flatten])
	net_a.set_input([x_a, x_a_overlap])

	pair_combine = PairCombine(shape1=kernals, shape2=kernals)
	pair_combine.set_input([net_q.output, net_a.output])

	hidden_layer = Linear(n_in=2*kernals+1, n_out=2*kernals+1)
	hidden_layer.set_input(pair_combine.output)
	
	lr_layer = LR(n_in=2*kernals+1, n_out = n_class)
	lr_layer.set_input(hidden_layer.output)

	#Cost
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lr_layer.output, labels=y)) 
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	#run
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		index = 1
		for train_q, overlap_q, train_a, overlap_a, label in Batcher([q_train, q_overlap_train, a_train, a_overlap_train, y_train], batch_size=batch_size):
			#result = sess.run(pair_combine.output, feed_dict={x_q:train_q, x_q_overlap:overlap_q, x_a:train_a, x_a_overlap:overlap_a})
			sess.run(optimizer, feed_dict={x_q:train_q, x_q_overlap:overlap_q, x_a:train_a, x_a_overlap:overlap_a, y:label})
			loss = sess.run(cost, feed_dict={x_q:q_dev[:batch_size], x_q_overlap:q_overlap_dev[:batch_size], x_a:a_dev[:batch_size], x_a_overlap:a_overlap_dev[:batch_size], y:y_dev[:batch_size]})
			print index, loss
			index += 1
			time.sleep(1)
