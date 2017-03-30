#! /bin/python
# -*- coding=utf8 -*-

import tensorflow as tf
import numpy
import os
import sys
import time
from getBatch import Batcher
import layer
import cPickle 

#LOAD
data_dir = '../TRAIN.bak'

q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
vocab_emb = numpy.load(os.path.join(data_dir, 'emb_vectors.bin.npy'))

#Load PRINT
print 'word embeddings ndim', vocab_emb.shape[1]

#Init
q_max_sent_size = q_train.shape[1]
a_max_sent_size = a_train.shape[1]
word_dim = vocab_emb.shape[1]
filter_width = 5
kernals = 100
batch_size = 50
overlap_ndim = 5
n_class = 2
learning_rate = 0.001
overlap_word_max_id = numpy.max(q_overlap_train)
filter_shape = [filter_width, word_dim + overlap_ndim, 1, kernals]
numpy_rng = numpy.random.RandomState(123)
vocab_emb_overlap = tf.Variable(numpy_rng.randn(overlap_word_max_id+1, overlap_ndim)*0.25, name='W')
t_vocab_emb_overlap = tf.random_normal(shape=[overlap_word_max_id+1, overlap_ndim])
x_q = tf.placeholder(tf.int32, [None, q_max_sent_size], name='x_q')
x_q_overlap = tf.placeholder(tf.int32, [None, q_max_sent_size], name='x_q_overlap')
x_a = tf.placeholder(tf.int32, [None, a_max_sent_size], name='x_a')
x_a_overlap = tf.placeholder(tf.int32, [None, a_max_sent_size], name='x_a_overlap')

#Define Net
lookup_words = layer.LookupTable(W=vocab_emb, batch_size=batch_size, filter_width=filter_width, width=word_dim)
lookup_words_overlap = layer.LookupTable(W=vocab_emb_overlap, batch_size=batch_size, filter_width=filter_width, width=overlap_ndim)
lookup_words = layer.ParallelLookupTable([lookup_words, lookup_words_overlap])
conv = layer.Conv2d(filter_shape=filter_shape, filter_width=filter_width, batch_size=batch_size, height=q_max_sent_size+2*(filter_width-1), width=overlap_ndim+word_dim)
non_line = layer.Activation(activation=tf.tanh, nkernals=kernals)
max_pool = layer.Maxpool(ksize=[1, q_max_sent_size+filter_width-1, 1, 1])
flatten = layer.Flatten(batch_size=batch_size)
net_q = layer.FeedForwardNet([lookup_words, conv, non_line, max_pool, flatten])
net_q.set_input([x_q, x_q_overlap])

lookup_words = layer.LookupTable(W=vocab_emb, batch_size=batch_size, filter_width=filter_width, width=word_dim)
lookup_words_overlap = layer.LookupTable(W=vocab_emb_overlap, batch_size=batch_size, filter_width=filter_width, width=overlap_ndim)
lookup_words = layer.ParallelLookupTable([lookup_words, lookup_words_overlap])
conv = layer.Conv2d(filter_shape=filter_shape, filter_width=filter_width, batch_size=batch_size, height=a_max_sent_size+2*(filter_width-1), width=overlap_ndim+word_dim)
non_line = layer.Activation(tf.tanh, nkernals=kernals)
max_pool = layer.Maxpool(ksize=[1, a_max_sent_size+filter_width-1, 1, 1])
flatten = layer.Flatten(batch_size=batch_size)
net_a = layer.FeedForwardNet([lookup_words, conv, non_line, max_pool, flatten])
net_a.set_input([x_a, x_a_overlap])

pair_combine = layer.PairCombine(shape1=kernals, shape2=kernals)
pair_combine.set_input([net_q.output, net_a.output])

hidden_layer = layer.Linear(n_in=2*kernals+1, n_out=2*kernals+1)
hidden_layer.set_input(pair_combine.output)

lr_layer = layer.LR(n_in=2*kernals+1, n_out = n_class)
lr_layer.set_input(hidden_layer.output)

#Save Graph
tf.add_to_collection('x_q', x_q)
tf.add_to_collection('x_q_overlap', x_q_overlap)
tf.add_to_collection('x_a', x_a)
tf.add_to_collection('x_a_overlap', x_a_overlap)
tf.add_to_collection('pred', lr_layer.output)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver()
	saver.export_meta_graph('qa.graph')
