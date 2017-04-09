from __future__ import print_function
import os
import os.path
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

SOURCE_SUBJECT = 3
TARGET_SUBJECT = 4
ADD_TARGET = False
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
#def channel_seperate(x):

last_weights = weight_variable([n_hidden, n_classes])
last_biases = weight_variable([n_classes])

'''
 	Model definition here
'''
x = tf.placeholder("float", [None, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
x_ = tf.transpose(x, [0, 3, 2, 1]) 
layer2 = apply_conv(x_, 5, 1, conv_multiplier)
layer3 = apply_conv(layer2, 5, conv_multiplier, conv_multiplier)
layer4 = apply_conv(layer3, 5, conv_multiplier, conv_multiplier)
layer5 = apply_conv(layer4, 5, conv_multiplier, conv_multiplier)
layer67 = apply_lstm(layer5, last_weights, last_biases, rnn_layers, keep_prob)
layerOut = layer67
result = tf.nn.softmax(layerOut)
cross_entropy = -tf.reduce_sum(y*tf.log(result))
all_vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name ]) * 0.0005
loss = tf.add(cross_entropy, lossL2)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(result,1), tf.argmax(y,1))
target_count = tf.reduce_sum(tf.cast( tf.equal( 
												tf.cast(tf.argmax(y,1), "float"), tf.zeros([batch_size]) 
												) 
									, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]

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

'''	
	load target subject test data
'''
raw_X_test_target, raw_y_test_target = load_test(TARGET_SUBJECT)
X_test_target, y_test_target = segment_signal(raw_X_test_target, raw_y_test_target)
X_test_target = np.expand_dims(X_test_target, 1)

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
'''
	Load target subject train data
'''
if(ADD_TARGET):
	if(os.path.exists(target_path+"_unlabelled_data.npy")):
		X_train_target = np.load(target_path+"_unlabelled_data.npy")
		y_train_target = np.full((X_train_target.shape[0],n_classes),0, dtype=float)
	else:
		raw_X_train_unlabelled, raw_y_train_unlabelled = load_train(TARGET_SUBJECT)
		X_train_target, y_train_target = segment_signal(raw_X_train_unlabelled, raw_y_train_unlabelled)
		X_train_target = np.expand_dims(X_train_target, 1)
		y_train_target = np.full((X_train_target.shape[0], n_classes),0,dtype=float)
		np.save(target_path+"_unlabelled_data", X_train_target)
	X_train = np.vstack((X_train, X_train_target))
	y_train = np.vstack((y_train, y_train_target))

print ("finish fetching")

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)


f1_lst = []
f1_tlst = []
#def main():
for iteration in range(100):
	#start_random = random.randint(0, batch_size)
	#start = start_random
	step = 0
	'''
		Train the model on all training data
	'''
	for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
		batch_x, batch_y = batch
		acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
		#unlabelled_rows_sum, correct_preds_sum = sess.run([target_count, tf.reduce_sum(tf.cast(correct_prediction, "float"))], feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
		#batch_y_ = np.argmax(batch_y, axis=1)
		#unlabelled_rows_sum = (batch_y_ == 0).sum()
		#acc = correct_preds_sum / (batch_size - unlabelled_rows_sum)
		if (acc < 0.7 and iteration > 20):
			sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
		if (acc < 0.8 + 0.0015 * iteration):
			sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
	'''
		Test the model on source target data
	'''

	test_pred = np.empty((0))
	test_true = np.empty((0))
	for batch in iterate_minibatches(X_test, y_test, batch_size):
		inputs, targets = batch
		acc, y_pred = sess.run([accuracy, tf.argmax(result, 1)], feed_dict = {x : inputs, y : targets, keep_prob : 1.0})
		test_pred = np.append(test_pred, y_pred, axis=0)
		test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)

	f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
	print("Iter" + str(iteration) + " Subject " + str(SOURCE_SUBJECT) + " Source Test F1 score= " + "{:.5f}".format(f1))
	f1_lst.append(f1)

	
	
	'''
		Test the model on target target data
	'''
	test_pred = np.empty((0))
	test_true = np.empty((0))
	for batch in iterate_minibatches(X_test_target, y_test_target, batch_size):
		inputs, targets = batch
		acc, y_pred = sess.run([accuracy, tf.argmax(result, 1)], feed_dict = {x : inputs, y : targets, keep_prob : 1.0})
		test_pred = np.append(test_pred, y_pred, axis=0)
		test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)

	f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
	print(" 	Subject " + str(TARGET_SUBJECT) + " Target Test F1 score= " + "{:.5f}".format(f1))
	f1_tlst.append(f1)
	
print (" Subject " + str(SOURCE_SUBJECT) + " best source f1 is " + str(max(f1_lst)))	
print (" Subject " + str(TARGET_SUBJECT) + " best target f1 is " + str(max(f1_tlst)))	
