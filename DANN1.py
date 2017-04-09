from __future__ import print_function
import os
import numpy as np
import math
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.contrib import rnn
import sklearn as sk
import cPickle as cp
from sliding_window import sliding_window
import random
from pre_define import *
from flip_gradient import *


SOURCE_SUBJECT = 3
TARGET_SUBJECT = 4
ADD_TARGET = True
data_folder = "./data/"
'''
	our model starts
'''

'''
	tensorflow wrapper functions 
'''
def weight_variable(shape):
	#initial = tf.orthogonal_initializer(shape)
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
	
def depthwise_conv2d(x, W):
	return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
	
def apply_depthwise_conv(x,kernel_size,in_channels,mul_channels):
	weights = weight_variable([1, kernel_size, in_channels, mul_channels])
	biases = bias_variable([in_channels * mul_channels])
	return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))

def apply_conv(x,kernel_size,in_channels,out_channels):
	weights = weight_variable([1, kernel_size, in_channels, out_channels])
	biases = bias_variable([out_channels])
	return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_lstm(x, weights, biases, rnn_layers, keep_prob):
	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, num_of_feature_map)
	# Required shape: 'SLIDING_WINDOW_LENGTH' tensors list of shape (batch_size, NB_SENSOR_CHANNELS * num_of_feature_map)
	# Permuting batch_size and SLIDING_WINDOW_LENGTH
	x = tf.transpose(x, [2, 0, 1, 3])
	# Reshaping to (SLIDING_WINDOW_LENGTH * batch_size,  NB_SENSOR_CHANNELS * num_of_feature_map)
	x = tf.reshape(x, [-1, NB_SENSOR_CHANNELS * conv_multiplier])
	# Split to get a list of 'SLIDING_WINDOW_LENGTH' tensors of shape (batch_size,  NB_SENSOR_CHANNELS * num_of_feature_map)
	x = tf.split(x, SLIDING_WINDOW_LENGTH - 4 * 4, 0)

	def lstm_cell():
	  return tf.contrib.rnn.BasicLSTMCell(
		  n_hidden, forget_bias=1.0, state_is_tuple=True)
	def attn_cell():
	  return tf.contrib.rnn.DropoutWrapper(
		  lstm_cell(), output_keep_prob=keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(rnn_layers)], state_is_tuple=True)

	outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return_value = tf.matmul(outputs[-1], weights) + biases

	return return_value

def apply_dense(x, input_size, out_channels):
	#x = tf.squeeze(x)
	x = tf.reshape(x, [-1, input_size])
	weights = weight_variable([input_size, out_channels])
	biases = bias_variable([out_channels])
	return tf.nn.relu(tf.add(tf.matmul(x, weights),biases))

'''
	Model definition here
'''
class DANN_Model(object):

	def __init__(self):
		self.build_model()


	def build_model(self):
		self.x = tf.placeholder("float", [None, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS])
		self.y = tf.placeholder("float", [None, n_classes])
		self.domain = tf.placeholder(tf.float32, [None, 2])
		self.keep_prob = tf.placeholder("float")
		self.l = tf.placeholder(tf.float32, [])
		self.train = tf.placeholder(tf.bool, [])
		x_ = tf.transpose(self.x, [0, 3, 2, 1])
		# CNN model for feature extraction
		with tf.variable_scope('feature_extractor'):
			layer2 = apply_conv(x_, 5, 1, conv_multiplier)
			layer3 = apply_conv(layer2, 5, conv_multiplier, conv_multiplier)
			layer4 = apply_conv(layer3, 5, conv_multiplier, conv_multiplier)
			layer5 = apply_conv(layer4, 5, conv_multiplier, conv_multiplier)
			self.feature = layer5
		# RNN + softmax for class prediction
		with tf.variable_scope('label_predictor'):
			#all_features = lambda: self.feature
			last_weights = weight_variable([n_hidden, n_classes])
			last_biases = weight_variable([n_classes])
			layer67 = apply_lstm(self.feature, last_weights, last_biases, rnn_layers, self.keep_prob)
			all_vars = tf.trainable_variables() 
			lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name ]) * 0.0005
			self.pred = tf.nn.softmax(layer67)
			self.pred_loss = tf.add(-tf.reduce_sum(self.y*tf.log(self.pred)), lossL2)
		# Small MLP for domain prediction with adversarial loss
		with tf.variable_scope('domain_predictor'):
			# Flip the gradient when backpropagating through this operation
			feat = flip_gradient(self.feature, self.l)
			layer_d1 = apply_dense(feat, conv_multiplier*(SLIDING_WINDOW_LENGTH - 4 * 4)*NB_SENSOR_CHANNELS, n_hidden)
			d_weights = weight_variable([n_hidden, 2])
			d_biases = bias_variable([2])
			layer_d2 = tf.matmul(layer_d1, d_weights) + d_biases
			self.domain_pred = tf.nn.softmax(layer_d2)
			self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=layer_d2, labels=self.domain)


# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
	model = DANN_Model()
	#learning_rate = tf.placeholder(tf.float32, [])
	pred_loss = tf.reduce_mean(model.pred_loss)
	domain_loss = tf.reduce_mean(model.domain_loss)
	total_loss = pred_loss + domain_loss
	regular_train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(pred_loss)
	dann_train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss)
	# Evaluation
	pred_labels = tf.argmax(model.pred, 1)
	correct_label_pred = tf.equal(tf.argmax(model.y, 1), tf.argmax(model.pred, 1))
	label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
	correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
	domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

def iterate_minibatches(inputs, targets, domains, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt], domains[excerpt]

'''
	Constructing train / test data here
'''
def data_loader(subject):
	raw_X_test, raw_y_test = load_test(subject)
	raw_X_train, raw_train_activity_labels = load_train(subject)
	X_train, y_train = segment_signal(raw_X_train, raw_train_activity_labels)
	X_train = np.expand_dims(X_train, 1)

	X_test, y_test = segment_signal(raw_X_test, raw_y_test)
	X_test = np.expand_dims(X_test, 1)
	return (X_train, y_train, X_test, y_test)

'''	
	load source subject test data
'''
raw_X_test, raw_y_test = load_test(SOURCE_SUBJECT)
X_test, y_test = segment_signal(raw_X_test, raw_y_test)
X_test = np.expand_dims(X_test, 1)
D_test = np.full((y_test.shape[0], 2), [1, 0])

'''	
	load target subject test data
'''
raw_X_test_target, raw_y_test_target = load_test(TARGET_SUBJECT)
X_test_target, y_test_target = segment_signal(raw_X_test_target, raw_y_test_target)
X_test_target = np.expand_dims(X_test_target, 1)
D_test_target = np.full((y_test_target.shape[0], 2), [0, 1])

'''
	Load source subject train data
'''
source_path = data_folder+"subject"+str(SOURCE_SUBJECT)
target_path = data_folder+"subject"+str(TARGET_SUBJECT)
if(os.path.exists(source_path+"_X_train.npy")):
	X_train = np.load(source_path+'_X_train.npy')
	y_train = np.load(source_path+'_y_train.npy')
else:
	raw_X_train, raw_train_activity_labels = load_train(SOURCE_SUBJECT)
	X_train, y_train = segment_signal(raw_X_train, raw_train_activity_labels)
	X_train = np.expand_dims(X_train, 1)
	np.save(source_path+"_X_train", X_train)
	np.save(source_path+"_y_train", y_train)
D_train = np.full((X_train.shape[0],2),[1,0],dtype=float)
'''
	Load target subject train data
'''
if(ADD_TARGET):
	if(os.path.exists(target_path+"_labelled_data.npy")):
		X_train_target = np.load(target_path+"_unlabelled_data.npy")
		y_train_target = np.full((X_train_target.shape[0],n_classes),0, dtype=float)
	else:
		raw_X_train_unlabelled, raw_y_train_unlabelled = load_train(TARGET_SUBJECT)
		X_train_target, y_train_target = segment_signal(raw_X_train_unlabelled, raw_y_train_unlabelled)
		X_train_target = np.expand_dims(X_train_target, 1)
		y_train_target = np.full((X_train_target.shape[0], n_classes),0,dtype=float)
		np.save(target_path+"_unlabelled_data", X_train_target)
	D_train_target = np.full((X_train_target.shape[0],2),[0,1],dtype=float)
	X_train = np.vstack((X_train, X_train_target))
	y_train = np.vstack((y_train, y_train_target))
	D_train = np.vstack((D_train, D_train_target))
print ("finish fetching")


# X_train/test shape : [num_windows, 1, SLIDING_WINDOW_LENGTH, in_channels]
# y_train/test shape : [num_windows, n_classes]
f1_lst = []
f1_tlst = []
def main():
	with tf.Session(graph=graph) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		for epoch in range(100):
			# Adaptation param and learning rate schedule as described in the paper
			p = float(epoch) / 100
			l = (2. / (1. + np.exp(-10. * p)) - 1)*3000
			'''
				Train the model on all training data
			'''
			for batch in iterate_minibatches(X_train, y_train, D_train, batch_size, shuffle=True):
				batch_x, batch_y, batch_d = batch
				correct_preds_sum = sess.run(tf.reduce_sum(tf.cast(correct_prediction, "float")), feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
				batch_y_ = np.argmax(batch_y, axis=1)
				unlabelled_rows_sum = (batch_y_ == 0).sum()
				acc = correct_preds_sum / (batch_size - unlabelled_rows_sum)
				if (acc < 0.7 and epoch > 20):
					sess.run(regular_train_op, feed_dict={model.x: batch_x, model.y: batch_y, model.domain : batch_d, model.l : l, model.keep_prob : 0.5})
				if (acc < 0.8 + 0.0015 * epoch):
					sess.run(regular_train_op, feed_dict={model.x: batch_x, model.y: batch_y, model.domain : batch_d, model.l : l, model.keep_prob : 0.5})
			'''
				Test the model on source target data
			'''

			test_pred = np.empty((0))
			test_true = np.empty((0))
			d_acc_sum = []
			for batch in iterate_minibatches(X_test, y_test, D_test, batch_size):
				inputs, targets, domains = batch
				y_pred, d_acc = sess.run([pred_labels, domain_acc], feed_dict = {model.x : inputs, model.y : targets, model.domain : domains, model.keep_prob : 1.0})
				test_pred = np.append(test_pred, y_pred, axis=0)
				test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)
				d_acc_sum.append(d_acc)
			d_acc_avg =  sum(d_acc_sum) / len(d_acc_sum)
			f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
			print("Iter" + str(epoch) + " Subject " + str(SOURCE_SUBJECT) + " Source Test F1 score= " + "{:.5f}".format(f1) + "		d_acc:"+str(d_acc_avg))
			f1_lst.append(f1)

			'''
				Test the model on target target data
			'''

			test_pred = np.empty((0))
			test_true = np.empty((0))
			d_acc_sum = []
			for batch in iterate_minibatches(X_test_target, y_test_target, D_test_target, batch_size):
				inputs, targets, domains = batch
				y_pred, d_acc = sess.run([pred_labels, domain_acc], feed_dict = {model.x : inputs, model.y : targets, model.domain : domains, model.keep_prob : 1.0})
				test_pred = np.append(test_pred, y_pred, axis=0)
				test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)
				d_acc_sum.append(d_acc)
			d_acc_avg =  sum(d_acc_sum) / len(d_acc_sum)
			f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
			print(" 	Subject " + str(TARGET_SUBJECT) + " Target Test F1 score= " + "{:.5f}".format(f1)+ "		d_acc:"+str(d_acc_avg))
			f1_tlst.append(f1)

	print (" Subject " + str(SOURCE_SUBJECT) + " best source f1 is " + str(max(f1_lst)))	
	print (" Subject " + str(TARGET_SUBJECT) + " best target f1 is " + str(max(f1_tlst)))


