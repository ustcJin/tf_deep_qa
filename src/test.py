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
	def __init__(self, W=None, batch_size=100, filter_width=5, width = 28):
		super(LookupTable, self).__init__()
		self.W = W
		self.filter_width = filter_width
		self.batch_size = batch_size
		self.width = width
		self.pad_matrix = numpy.zeros([self.batch_size, self.filter_width - 1, self.width])
	def output_func(self, input):
		out = tf.concat([self.pad_matrix, tf.to_float(tf.nn.embedding_lookup(self.W, input)), self.pad_matrix], 1)
		return out

class Conv2d(Layer):
	def __init__(self, filter_shape, filter_width = 5, batch_size = 100, height = 28, width = 28, nkernals = 1):
		bound = numpy.sqrt(1. / numpy.prod(filter_shape[0:2]))
		self.filter = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-bound, maxval=bound))
		self.filter_width = filter_width
		self.batch_size = batch_size
		self.height = height
		self.width = width
		self.nkernals = nkernals
	def output_func(self, input):
		input = tf.reshape(input, [-1, self.height, self.width, self.nkernals])
		return tf.nn.conv2d(input, self.filter, strides = [1,1,1,1], padding = 'VALID')

class Activation(Layer):
	def __init__(self, activation = tf.tanh, nkernals = 100):
		self.activation = activation
		self.bias = tf.Variable(tf.zeros(shape=[nkernals]))
		self.nkernals = nkernals
	def output_func(self, input):
		return self.activation(input +  tf.reshape(self.bias, [1,1,1,self.nkernals]))

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
		return tf.reshape(input, shape=[self.batch_size, 1, -1])

class PairCombine(Layer):
	def __init__(self, shape1, shape2):
		self.shape1 = shape1
		self.shape2 = shape2
		self.W = tf.Variable(tf.zeros(shape=[shape1, shape2]))
	def output_func(self, input):
		q, a= input[0], input[1]
		mid1 = tf.reshape(a, shape=[-1, self.shape1])
		mid2 = tf.matmul(mid1, self.W)
		mid3 = tf.reshape(mid2, shape=[-1, 1, self.shape2])
		sim = tf.matmul(q, mid3, transpose_b=True)
		return tf.concat([q, sim, a], 2);

class Linear(Layer):
	def __init__(self, n_in, n_out, activation=tf.tanh):
		bound=numpy.sqrt(1.2/(n_in+n_out))
		self.W = tf.Variable(tf.random_uniform(shape=[n_in, n_out], minval=-bound, maxval=bound))
		self.b = tf.Variable(tf.zeros(shape=[n_out]))
		self.activation = tf.tanh
		self.n_in = n_in
		self.n_out = n_out
	def output_func(self, input):
		mid1 = tf.reshape(input, [-1, self.n_in])
		mid2 = tf.add(tf.matmul(mid1, self.W), self.b)
		mid3 = tf.reshape(mid2, shape=[-1, self.n_out])
		#return mid3
		return self.activation(mid3)

class LR(Layer):
	def __init__(self, n_in, n_out):
		self.W = tf.Variable(tf.zeros(shape=[n_in, n_out]))
		self.b = tf.Variable(tf.zeros(shape=[n_out]))
		self.n_in = n_in
		self.n_out = n_out
	def output_func(self, input):
		mid1 = tf.reshape(input, [-1, self.n_in])
		mid2 = tf.add(tf.matmul(mid1, self.W), self.b)
		mid3 = tf.reshape(mid2, shape=[-1, self.n_out])
		return mid3

if __name__ == '__main__':
	#LOAD
	data_dir = '../TRAIN.bak'

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
	kernals = 100
	batch_size = 100
	overlap_ndim = 5
	n_class = 2
	learning_rate = 0.001
	overlap_word_max_id = numpy.max(q_overlap_train)
	filter_shape = [filter_width, word_dim + overlap_ndim, 1, kernals]
	numpy_rng = numpy.random.RandomState(123)
	#vocab_emb_overlap = numpy_rng.randn(overlap_word_max_id+1, overlap_ndim) * 0.25
	vocab_emb_overlap = tf.Variable(numpy_rng.randn(overlap_word_max_id+1, overlap_ndim)*0.25)
	t_vocab_emb_overlap = tf.random_normal(shape=[overlap_word_max_id+1, overlap_ndim])
	x_q = tf.placeholder(tf.int32, [None, q_max_sent_size])
	x_q_overlap = tf.placeholder(tf.int32, [None, q_max_sent_size])
	x_a = tf.placeholder(tf.int32, [None, a_max_sent_size])
	x_a_overlap = tf.placeholder(tf.int32, [None, a_max_sent_size])
	y = tf.placeholder(tf.int32, [None, n_class])

	#Define Net
	params = numpy.load('params.npy')
	#vocab_emb_overlap, net_q.layers[1].filter, net_q.layers[2].bias, net_a.layers[1].filter, net_a.layers[2].bias, pair_combine.W, hidden_layer.W, hidden_layer.b, lr_layer.W, lr_layer.b = params
	lookup_words = LookupTable(W=vocab_emb, batch_size=batch_size, filter_width=filter_width, width=word_dim)
	lookup_words_overlap = LookupTable(W=vocab_emb_overlap, batch_size=batch_size, filter_width=filter_width, width=overlap_ndim)
	lookup_words_overlap.W = tf.assign(lookup_words_overlap.W, params[0])
	lookup_words = ParallelLookupTable([lookup_words, lookup_words_overlap])
	conv = Conv2d(filter_shape=filter_shape, filter_width=filter_width, batch_size=batch_size, height=q_max_sent_size+2*(filter_width-1), width=overlap_ndim+word_dim)
	conv.filter = tf.assign(conv.filter, params[-9])
	non_line = Activation(activation=tf.tanh, nkernals=kernals)
	non_line.bias = tf.assign(non_line.bias, params[-8])
	max_pool = Maxpool(ksize=[1, q_max_sent_size+filter_width-1, 1, 1])
	flatten = Flatten(batch_size=batch_size)
	net_q = FeedForwardNet([lookup_words, conv, non_line, max_pool, flatten])
	net_q.set_input([x_q, x_q_overlap])

	lookup_words = LookupTable(W=vocab_emb, batch_size=batch_size, filter_width=filter_width, width=word_dim)
	lookup_words_overlap = LookupTable(W=vocab_emb_overlap, batch_size=batch_size, filter_width=filter_width, width=overlap_ndim)
	lookup_words_overlap.W = tf.assign(lookup_words_overlap.W, params[0])
	lookup_words = ParallelLookupTable([lookup_words, lookup_words_overlap])
	conv = Conv2d(filter_shape=filter_shape, filter_width=filter_width, batch_size=batch_size, height=a_max_sent_size+2*(filter_width-1), width=overlap_ndim+word_dim)
	conv.filter = tf.assign(conv.filter, params[-7])
	non_line = Activation(tf.tanh, nkernals=kernals)
	non_line.bias = tf.assign(non_line.bias, params[-6])
	max_pool = Maxpool(ksize=[1, a_max_sent_size+filter_width-1, 1, 1])
	flatten = Flatten(batch_size=batch_size)
	net_a = FeedForwardNet([lookup_words, conv, non_line, max_pool, flatten])
	net_a.set_input([x_a, x_a_overlap])

	pair_combine = PairCombine(shape1=kernals, shape2=kernals)
	pair_combine.W = tf.assign(pair_combine.W, params[-5])
	pair_combine.set_input([net_q.output, net_a.output])

	hidden_layer = Linear(n_in=2*kernals+1, n_out=2*kernals+1)
	hidden_layer.W = tf.assign(hidden_layer.W, params[-4])
	hidden_layer.b = tf.assign(hidden_layer.b, params[-3])
	hidden_layer.set_input(pair_combine.output)

	
	lr_layer = LR(n_in=2*kernals+1, n_out = n_class)
	lr_layer.W = tf.assign(lr_layer.W, params[-2])
	lr_layer.b = tf.assign(lr_layer.b, params[-1])
	lr_layer.set_input(hidden_layer.output)

	train_net = FeedForwardNet([net_q, net_a, pair_combine, hidden_layer, lr_layer])
	

	#run
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		index = 1
		epoch = 0
		while epoch < 1:
			right_cnt = 0
			for train_q, overlap_q, train_a, overlap_a, label in Batcher([q_dev, q_overlap_dev, a_dev, a_overlap_dev, y_dev], batch_size=batch_size):
				result = sess.run(train_net.layers[-1].output, feed_dict={x_q:train_q, x_q_overlap:overlap_q, x_a:train_a, x_a_overlap:overlap_a, y:label})
				for i in range(batch_size):
					s = result[i][0] * label[i][0] + result[i][1] * label[i][1]
					if s > 0:
						right_cnt += 1	
			epoch += 1
			print 'right count', right_cnt, len(y_dev)
