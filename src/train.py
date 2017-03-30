#! /bin/python
# -*- coding=utf8 -*-

import tensorflow as tf
import numpy
import os
import sys
import time
from getBatch import Batcher
import layer

#LOAD
data_dir = '../TRAIN.bak'

q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))
learning_rate = 0.001
batch_size = 50
n_class = 2
y_label = tf.placeholder(tf.int32, [None, n_class])

#Load graph
new_saver = tf.train.import_meta_graph('qa.graph')
x_q = tf.get_collection('x_q')[0]
x_q_overlap = tf.get_collection('x_q_overlap')[0]
x_a = tf.get_collection('x_a')[0]
x_a_overlap = tf.get_collection('x_a_overlap')[0]
pred = tf.get_collection('pred')[0]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_label))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#run
epoches = 15
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	index = 1
	epoch = 0
	while epoch < epoches:
		for train_q, overlap_q, train_a, overlap_a, label in Batcher([q_train, q_overlap_train, a_train, a_overlap_train, y_train], batch_size=batch_size):
			sess.run(optimizer, feed_dict={x_q:train_q, x_q_overlap:overlap_q, x_a:train_a, x_a_overlap:overlap_a, y_label:label})
		epoch += 1
		print epoch
	print 'trainning done.'
	saver = tf.train.Saver()
	saver.save(sess, 'qa.params')
