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

data_path = os.getcwd() + "/../output1/"

## Parameters
global_step = tf.Variable(0, trainable=False)
#base_learning_rate = 0.0001
#learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,
#                                           150, 0.9)
learning_rate = 0.0004
batch_size = 100
display_step = 15
# Network Parameters
SLIDING_WINDOW_LENGTH = 24 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 18 # Activity label, including null class
rnn_layers = 2 # Number of rnn layers
conv_multiplier = 64
# Session Parameters
SLIDING_WINDOW_STEP = 12
# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113


'''
	funtions for get raw data
'''
def load_dataset(filename):

	f = file(filename, 'rb')
	data = cp.load(f)
	f.close()

	X_train, y_train = data[0]
	X_test, y_test = data[1]

	print(" ..from file {}".format(filename))
	print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)

	# The targets are casted to int8 for GPU compatibility.
	y_train = y_train.astype(np.uint8)
	y_test = y_test.astype(np.uint8)

	return X_train, y_train, X_test, y_test

def load_all(i):
	data = np.load(data_path+'S'+str(i)+'-Drill.dat_data.npy')
	labels = np.load(data_path+'S'+str(i)+'-Drill.dat_activity_labels.npy')
	for j in [1, 2, 3]:
		data = np.vstack((data, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_data.npy')))
		labels = np.vstack((labels, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_activity_labels.npy')))
	return (data, labels)

def load_test(i):
	data = np.load(data_path+'S'+str(i)+'-ADL4.dat_data.npy')
	labels = np.load(data_path+'S'+str(i)+'-ADL4.dat_activity_labels.npy')
	data = np.vstack((data, np.load(data_path+'S'+str(i)+'-ADL5.dat_data.npy')))
	labels = np.vstack((labels, np.load(data_path+'S'+str(i)+'-ADL5.dat_activity_labels.npy')))
	'''
	data = np.vstack((data, np.load(data_path+'S'+str(j)+'-ADL4.dat_data.npy')))
	labels = np.vstack((labels, np.load(data_path+'S'+str(j)+'-ADL4.dat_activity_labels.npy')))
	data = np.vstack((data, np.load(data_path+'S'+str(j)+'-ADL5.dat_data.npy')))
	labels = np.vstack((labels, np.load(data_path+'S'+str(j)+'-ADL5.dat_activity_labels.npy')))
	'''
	return (data, labels)

''' 
	functions for segementing data
'''
def windows(data, size):
	start = 0
	while start + size < data.shape[0]:
		yield start, start + size
		start += SLIDING_WINDOW_STEP
		
def segment_signal(data, activity_labels):
	segments = np.empty((0, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
	labels = np.empty((0, n_classes))
	for (start, end) in windows(data, SLIDING_WINDOW_LENGTH):
		segments = np.vstack([segments, np.array([data[start : end]])])
		labels = np.vstack((labels, activity_labels[end]))
	return segments, labels

def opp_sliding_window(data_x, data_y, ws, ss):
	data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
	data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
	return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def ReshapeActivityLabels(labels):
	#These are label unique indexes 
	new_labels = np.zeros(shape=(labels.shape[0], 18))
	for i in range(labels.shape[0]):
		new_labels[i][labels[i]] = 1
	return new_labels

def trim_null(X, y):
	assert X.shape[0] == y.shape[0]
	nullset = []
	for i in range(y.shape[0]):
		if(y[i] == 0):
			nullset.append(i)
	X_trim = np.delete(X, nullset, axis=0)
	y_trim = np.delete(y, nullset, axis=0)
	return (X_trim, y_trim)

raw_X_test, raw_y_test = load_test(1)
raw_X_test1, raw_y_test1 = load_test(2)

raw_X_train, raw_train_activity_labels = load_all(1)
print ("finish fetching")

# X_train : [num_windows, 1, SLIDING_WINDOW_LENGTH, in_channels]
# y_train : [num_windows, n_classes]
X_train, y_train = segment_signal(raw_X_train, raw_train_activity_labels)
X_train = np.expand_dims(X_train, 1)
#X_train = np.load(data_path+"X_train1_123D.npy")
#y_train = np.load(data_path+"y_train_123D.npy")

X_test, y_test = segment_signal(raw_X_test, raw_y_test)
X_test = np.expand_dims(X_test, 1)

X_test1, y_test1 = segment_signal(raw_X_test1, raw_y_test1)
X_test1 = np.expand_dims(X_test1, 1)
#X_test = np.load(data_path+"X_test_23_45.npy")
#y_test = np.load(data_path+"y_test_23_45.npy")

#X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')
#assert NB_SENSOR_CHANNELS == X_train.shape[1]
#X_train, y_train = trim_null(X_train, y_train)
#X_test, y_test = trim_null(X_test, y_test)
# Sensor data is segmented using a sliding window mechanism
#X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
#X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
# Data is reshaped since the input of the network is a 4 dimension tensor
#X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
#X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
#y_train = ReshapeActivityLabels(y_train)
#y_test = ReshapeActivityLabels(y_test)
'''
	our model starts
'''

'''
	tensorflow wrapper functions for constructing CNN
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


x = tf.placeholder("float", [None, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")
#_x_ = tf.variable(tf.empty())
x_ = tf.transpose(x, [0, 3, 2, 1])
#x_ shape [batch_size, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, 1]
#x_ = tf.split(x_, NB_SENSOR_CHANNELS)
#for 
layer2 = apply_conv(x_, 5, 1, conv_multiplier)
layer3 = apply_conv(layer2, 5, conv_multiplier, conv_multiplier)
layer4 = apply_conv(layer3, 5, conv_multiplier, conv_multiplier)
layer5 = apply_conv(layer4, 5, conv_multiplier, conv_multiplier)


#layer2 = apply_depthwise_conv(x, 5, NB_SENSOR_CHANNELS, conv_multiplier)
#layer3 = apply_conv(layer2, 5, conv_multiplier * NB_SENSOR_CHANNELS, conv_multiplier * NB_SENSOR_CHANNELS)
#layer4 = apply_depthwise_conv(layer3, 5, conv_multiplier * NB_SENSOR_CHANNELS * 2, 2)
#layer5 = apply_depthwise_conv(layer4, 5, conv_multiplier * NB_SENSOR_CHANNELS * 4, 2)
#layer5 = layer3
#layer6_Base = apply_dense(layer5, 1 * conv_multiplier * NB_SENSOR_CHANNELS * (SLIDING_WINDOW_LENGTH - 4 * 4), 128)
#layer7_Base = apply_dense(layer6_Base, 128, 128)
#layer7_Base = layer6_Base
#layerOut = tf.matmul(layer7_Base, last_weights) + last_biases



layer67 = apply_lstm(layer5, last_weights, last_biases, rnn_layers, keep_prob)
layerOut = layer67
result = tf.nn.softmax(layerOut)
cross_entropy = -tf.reduce_sum(y*tf.log(result))
all_vars   = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name ]) * 0.0005
loss = tf.add(cross_entropy, lossL2)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(result,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.InteractiveSession()
sess.run(init)

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

def main():
	for iteration in range(100):
		#start_random = random.randint(0, batch_size)
		#start = start_random
		step = 0
		for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
			batch_x, batch_y = batch
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
			if (acc < 0.7 and iteration > 20):
				sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
			if (acc < 0.8 + 0.0015 * iteration):
				sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5})
			'''
			if (step % 100 == 0):
				# Calculate batch accuracy
				acc, preds = sess.run([accuracy, tf.argmax(result, 1)], feed_dict={x: batch_x, y: batch_y, keep_prob : 1.0})
				global_step_now, learning_rate_now = sess.run([global_step, learning_rate])
				print ("step: " + str(global_step_now) + " learning-rate: " + str(learning_rate_now))
				#for debugging
				dis = [0] * n_classes
				y_ = np.argmax(batch_y, axis = 1)
				for _y in y_:
					dis[_y] += 1
				print (dis)
				print (preds)
				print(str(step) + ": " + "Iter " + str(iteration) +  ", Training Accuracy= " + \
					  "{:.5f}".format(acc))
			step += 1
			'''
		
		# Classification of the testing data
		#print("Testing: Processing {0} instances in mini-batches of {1}".format(X_test.shape[0], batch_size))
		test_pred = np.empty((0))
		test_true = np.empty((0))
		#tf_confusion_metrics(result, y_train[start:], sess, {x : X_train[start:]})
		for batch in iterate_minibatches(X_train, y_train, batch_size):
			inputs, targets = batch
			acc, y_pred = sess.run([accuracy, tf.argmax(result, 1)], feed_dict = {x : inputs, y : targets, keep_prob : 1.0})
			test_pred = np.append(test_pred, y_pred, axis=0)
			test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)

		f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
		print("Iter " + str(iteration) + " Test F1 score= " + "{:.5f}".format(f1))
		test_pred = np.empty((0))
		test_true = np.empty((0))
		#tf_confusion_metrics(result, y_train[start:], sess, {x : X_train[start:]})
		for batch in iterate_minibatches(X_test1, y_test1, batch_size):
			inputs, targets = batch
			acc, y_pred = sess.run([accuracy, tf.argmax(result, 1)], feed_dict = {x : inputs, y : targets, keep_prob : 1.0})
			test_pred = np.append(test_pred, y_pred, axis=0)
			test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)

		f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
		print("Iter " + str(iteration) + " Test1 F1 score= " + "{:.5f}".format(f1))

def max(data):
	max_item = 0
	for i in data:
		if i > max_item:
			max_item = i
	return i



