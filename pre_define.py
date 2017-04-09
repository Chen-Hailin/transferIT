from __future__ import print_function
import os
import numpy as np
import math
import cPickle as cp
from sliding_window import sliding_window
import random

data_path = os.getcwd() + "/../output1/"
## Parameters
learning_rate = 0.0001
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
def load_adl1(i):
	data = np.empty([0,NB_SENSOR_CHANNELS])
	labels = np.empty([0, n_classes])
	for j in [1]:
		data = np.vstack((data, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_data.npy')))
		labels = np.vstack((labels, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_activity_labels.npy')))
	return (data, labels)

def load_adl(i, k):
	data = np.empty([0,NB_SENSOR_CHANNELS])
	labels = np.empty([0, n_classes])
	for j in [k]:
		data = np.vstack((data, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_data.npy')))
		labels = np.vstack((labels, np.load(data_path+'S'+str(i)+'-ADL'+str(j)+'.dat_activity_labels.npy')))
	return (data, labels)

def load_train(i):
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

def distribution(y):
	dis = [0] * n_classes
	y_ = np.argmax(y, axis = 1)
	for _y in y_:
		dis[_y] += 1
	return dis

def check_dis(y):
	dis = distribution(y)
	sum_dis = float(sum(dis))
	dis_per = [v / sum_dis for v in dis]
	print ('\n'.join('activity{}: {:0.2f}'.format(v, i) for v, i in enumerate(dis_per)))

def shrink(x, y, factor):
	assert x.shape[0] == y.shape[0]
	assert factor <= 1
	lst = np.arange(x.shape[0])
	y_ = np.argmax(y, axis = 1)
	np.random.shuffle(lst)
	dis = distribution(y)
	dis_shrink = [int(v*factor + 0.5) for v in dis]
	x_shrink = np.empty([0,x.shape[1],x.shape[2]])	
	y_shrink = np.empty([0,y.shape[1]])
	for i in lst:
		if dis_shrink[y_[i]] > 0:
			x_shrink = np.vstack((x_shrink, [x[i]]))
			y_shrink = np.vstack((y_shrink, [y[i]]))
			dis_shrink[y_[i]] -= 1
	assert sum(dis_shrink) == 0
	return (x_shrink, y_shrink)

def random_shrink(x, y, factor):
	assert x.shape[0] == y.shape[0]
	assert factor <= 1
	lst = np.arange(x.shape[0])
	y_ = np.argmax(y, axis = 1)
	np.random.shuffle(lst)
	dis = distribution(y)
	total = int(sum(dis) * factor)
	x_shrink = np.empty([0,x.shape[1],x.shape[2]])	
	y_shrink = np.empty([0,y.shape[1]])
	for i in range (total):
		index = lst[i]
		x_shrink = np.vstack((x_shrink, [x[index]]))
		y_shrink = np.vstack((y_shrink, [y[index]]))
	assert x_shrink.shape[0] == total
	return (x_shrink, y_shrink)

def shrink1(x, y):
	assert x.shape[0] == y.shape[0]
	lst = np.arange(x.shape[0])
	y_ = np.argmax(y, axis = 1)
	np.random.shuffle(lst)
	dis_shrink = [1] * y.shape[1]
	x_shrink = np.empty([0,x.shape[1],x.shape[2]])	
	y_shrink = np.empty([0,y.shape[1]])
	for i in lst:
		if dis_shrink[y_[i]] > 0:
			x_shrink = np.vstack((x_shrink, [x[i]]))
			y_shrink = np.vstack((y_shrink, [y[i]]))
			dis_shrink[y_[i]] -= 1
	assert sum(dis_shrink) == 1 and x_shrink.shape[0] == n_classes-1
	return (x_shrink, y_shrink)
