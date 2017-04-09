from __future__ import print_function
import os
import numpy as np
import math
import cPickle as cp
from sliding_window import sliding_window
import random
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
