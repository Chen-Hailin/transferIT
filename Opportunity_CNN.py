from __future__ import print_function
import os
import numpy as np
import math
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.contrib import rnn

data_path = os.getcwd() + "/output1/"
def load_all(i):
	data = np.load(data_path+'S'+str(i)+'-Drill.dat_data.npy')
	labels = np.load(data_path+'S'+str(i)+'-Drill.dat_activity_labels.npy')
	for j in range (1, 6):
		data = np.vstack((data, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_data.npy')))
		labels = np.vstack((labels, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_activity_labels.npy')))
	return (data, labels)
#raw_train_data = np.load(data_path + 'S1-Drill.dat_data.npy')
#raw_test_activity_labels = np.load(data_path + 'S1-Drill.dat_activity_labels.npy')
#raw_train_data, raw_test_activity_labels = load_all(1)


## Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
training_iters = 100000
# Network Parameters
input_channel = 133 # opportunity data input (img shape: 28*28)
window_size = 24 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 18 # Activity label, including null class
rnn_layers = 1 # Number of rnn layers
conv_multiplier = 64
# Session Parameters
sliding_step_size = 5   
'''
	functions for segementing data
'''
def windows(data, size):
	start = 0
	while start + size < data.shape[0]:
		yield start, start + size
		start += sliding_step_size
		
def segment_signal(data, activity_labels):
	segments = np.empty((0, window_size, input_channel))
	labels = np.empty((0, n_classes))
	for (start, end) in windows(data, window_size):
		segments = np.vstack([segments, np.array([data[start : end]])])
		labels = np.vstack((labels, activity_labels[end]))
	return segments, labels

'''
	tensorflow wrapper functions for constructing CNN
'''
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
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

def apply_lstm(x, weights, biases, rnn_layers):
	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, input_channel1, window_size, input_channel2)
	# Required shape: 'window_size' tensors list of shape (batch_size, input_channel12)
	# Permuting batch_size and window_size
	x = tf.transpose(x, [2, 0, 1, 3])
	# Reshaping to (window_size * batch_size, input_channel12)
	x = tf.reshape(x, [-1, input_channel * conv_multiplier])
	# Split to get a list of 'window_size' tensors of shape (batch_size, input_channel12)
	x = tf.split(x, window_size - 4 * 4, 0)

	def lstm_cell():
	  return tf.contrib.rnn.BasicLSTMCell(
		  n_hidden, forget_bias=1.0, state_is_tuple=True)

	cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(rnn_layers)], state_is_tuple=True)

	outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return_value = tf.matmul(outputs[-1], weights) + biases

	return return_value

def apply_dense(x, input_size, out_channels):
	#x = tf.squeeze(x)
	x = tf.reshape(x, [batch_size, -1])
	weights = weight_variable([input_size, out_channels])
	biases = bias_variable([out_channels])
	return tf.nn.relu(tf.add(tf.matmul(x, weights),biases))
#def channel_seperate(x):

last_weights = weight_variable([n_hidden, n_classes])
last_biases = weight_variable([n_classes])

# train_data : [num_windows, 1, window_size, in_channels]
# train_labels : [num_windows, n_classes]
#train_data, train_labels = segment_signal(raw_train_data, raw_test_activity_labels) 
#train_data = np.expand_dims(train_data, 1)
train_data = np.load(data_path+"train_data.npy")
train_labels = np.load(data_path+"train_labels.npy")


x = tf.placeholder("float", [None, 1, window_size, input_channel])
y = tf.placeholder("float", [None, n_classes])
#_x_ = tf.variable(tf.empty())
x_ = tf.transpose(x, [0, 3, 2, 1])
#x_ = tf.split(x_, input_channel)
#for 
layer2 = apply_conv(x_, 5, 1, conv_multiplier)
layer3 = apply_conv(layer2, 5, conv_multiplier, conv_multiplier)
layer4 = apply_conv(layer3, 5, conv_multiplier, conv_multiplier)
layer5 = apply_conv(layer4, 5, conv_multiplier, conv_multiplier)


#layer2 = apply_depthwise_conv(x, 5, input_channel, conv_multiplier)
#layer3 = apply_conv(layer2, 5, conv_multiplier * input_channel, conv_multiplier * input_channel)
#layer4 = apply_depthwise_conv(layer3, 5, conv_multiplier * input_channel * 2, 2)
#layer5 = apply_depthwise_conv(layer4, 5, conv_multiplier * input_channel * 4, 2)
#layer5 = layer3
layer6_Base = apply_dense(layer5, 1 * conv_multiplier * input_channel * (window_size - 4 * 4), 128)
layer7_Base = apply_dense(layer6_Base, 128, 128)
#layer7_Base = layer6_Base
layerOut = tf.matmul(layer7_Base, last_weights) + last_biases



#layer67 = apply_lstm(layer5, last_weights, last_biases, 2)
#layerOut = layer67
result = tf.nn.softmax(layerOut)
cross_entropy = -tf.reduce_sum(y*tf.log(result))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(result,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.InteractiveSession()
sess.run(init)

def main():
	start = 0
	while start + batch_size < train_data.shape[0]:
		batch_x = train_data[start : start + batch_size]
		batch_y = train_labels[start : start + batch_size]
		sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
		if start % (display_step * batch_size) == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			preds = sess.run(tf.argmax(result, 1), feed_dict={x: batch_x})
			
			#for debugging
			if (acc == 0 or acc > 0.3 or 1):
				dis = [0] * n_classes
				y_ = np.argmax(batch_y, axis = 1)
				for _y in y_:
					dis[_y] += 1
				print (dis)
				print (preds)

			print("Iter " + str(start) +  ", Training Accuracy= " + \
				  "{:.5f}".format(acc))
		start += batch_size

def max(data):
	max_item = 0
	for i in data:
		if i > max_item:
			max_item = i
	return i



