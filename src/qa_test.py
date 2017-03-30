#! /bin/python
# -*- coding=utf8 -*-

import tensorflow as tf
import numpy
import os
import sys
import time
from getBatch import Batcher

#INIT
data_dir = '../TRAIN.bak'

q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
batch_size = 50
n_class = 2
y_label = tf.placeholder(tf.int32, [None, n_class])

#load graph and param
new_saver = tf.train.import_meta_graph('qa.graph')
x_q = tf.get_collection('x_q')[0]
x_q_overlap = tf.get_collection('x_q_overlap')[0]
x_a = tf.get_collection('x_a')[0]
x_a_overlap = tf.get_collection('x_a_overlap')[0]
pred = tf.get_collection('pred')[0]

#run
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	new_saver.restore(sess, './qa.params')
	right_cnt = 0
	for test_q, overlap_q, test_a, overlap_a, label in Batcher([q_test, q_overlap_test, a_test, a_overlap_test, y_test], batch_size=batch_size):
		result = sess.run(pred, feed_dict={x_q:test_q, x_q_overlap:overlap_q, x_a:test_a, x_a_overlap:overlap_a, y_label:label})
		for i in range(batch_size):
			s = result[i][0] * label[i][0] + result[i][1] * label[i][1]
			if s > 0:
				right_cnt += 1
	print 'right ', right_cnt, float(right_cnt)/len(y_test)
