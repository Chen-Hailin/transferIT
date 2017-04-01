__author__ = 'fjordonez'  
import lasagne 
import theano 
import time  
import numpy as np 
import cPickle as cp 
import theano.tensor as T 
from sliding_window import sliding_window 
from sklearn.metrics import f1_score  

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge 
NB_SENSOR_CHANNELS = 113  

# Hardcoded number of classes in the gesture recognition problem 
NUM_CLASSES = 18  

# Hardcoded length of the sliding window mechanism employed to segment the data 
SLIDING_WINDOW_LENGTH = 24  
# Length of the input sequence after convolutional operations 
FINAL_SEQUENCE_LENGTH = 8  
# Hardcoded step of the sliding window mechanism employed to segment the data 
SLIDING_WINDOW_STEP = 12  
# Batch Size 
BATCH_SIZE = 100  
# Number filters convolutional layers 
NUM_FILTERS = 64  
# Size filters convolutional layers 
FILTER_SIZE = 5  
# Number of unit in the long short-term recurrent layers 
NUM_UNITS_LSTM = 128  
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

def trim_null(X, y):
	assert X.shape[0] == y.shape[0]
	nullset = []
	for i in range(y.shape[0]):
		if(y[i] == 0):
			nullset.append(i)
	X_trim = np.delete(X, nullset, axis=0)
	y_trim = np.delete(y, nullset, axis=0)
	return (X_trim, y_trim)
print("Loading data...") 
X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_gestures.data')  
assert NB_SENSOR_CHANNELS == X_train.shape[1] 
X_train, y_train = trim_null(X_train, y_train)
X_test, y_test = trim_null(X_test, y_test)
def opp_sliding_window(data_x, data_y, ws, ss): 
	data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
	data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)]) 
	return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)  

# Sensor data is segmented using a sliding window mechanism
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP) 
print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape)) 
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP) 
print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))  
# Data is reshaped since the input of the network is a 4 dimension tensor 
X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)) 
X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  
# Creating the Lasagne network representing the DeepConvLSTM architecture 
input_var = T.tensor4('inputs') 
target_var = T.ivector('targets') 
net = {} 
net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS), input_var=input_var) 
net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1)) 
net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1)) 
net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1)) 
net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1)) 
net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3)) 
net['lstm1'] = lasagne.layers.LSTMLayer(lasagne.layers.dropout(net['shuff'], p=.5), NUM_UNITS_LSTM) 
net['lstm2'] = lasagne.layers.LSTMLayer(lasagne.layers.dropout(net['lstm1'], p=.5), NUM_UNITS_LSTM) 
# In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions 
# to cause each time step of each sequence to be processed independently (see Lasagne docs for further information) 
net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM)) 
net['prob'] = lasagne.layers.DenseLayer(lasagne.layers.dropout(net['shp1'], p=.5),NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax) 
# Tensors reshaped back to the original shape 
net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES)) 
# Last sample in the sequence is considered 
net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)  
print("Compiling...") 
train_prediction_rnn = lasagne.layers.get_output(net['output'], input_var, deterministic=False) 
train_loss_rnn = lasagne.objectives.categorical_crossentropy(train_prediction_rnn, target_var) 
train_loss_rnn = train_loss_rnn.mean() 
train_loss_rnn += .0001 * lasagne.regularization.regularize_network_params(net['output'], lasagne.regularization.l2) 
params_rnn = lasagne.layers.get_all_params(net['output'], trainable=True) 
lr=0.0001 
rho=0.9 
updates_rnn = lasagne.updates.rmsprop(train_loss_rnn, params_rnn, learning_rate=lr, rho=rho) 
train_fn_rnn = theano.function([input_var, target_var], train_loss_rnn, updates=updates_rnn)  
test_prediction = lasagne.layers.get_output(net['output'], input_var, deterministic=True) 
test_loss_rnn = lasagne.objectives.categorical_crossentropy(test_prediction, target_var) 
test_loss_rnn = test_loss_rnn.mean() 
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype='float32')
test_fn_rnn = theano.function([input_var, target_var], [test_loss_rnn, T.argmax(test_prediction, axis=1), test_acc])  
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

tr_l_rnn = [] 
ts_l_rnn = [] 
iteration = 0
try: 
	num_epochs = 100 
	for epoch in range(num_epochs): 
		print("\tEpoch {}. Free gpu memory: {} MiB".format((epoch + 1),theano.sandbox.cuda.mem_info()[0] / 1024**2))  
		train_err, train_batches = 0, 0 
		start_time = time.time() 
		for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True): 
			inputs, targets = batch 
			train_err += train_fn_rnn(inputs, targets) 
			train_batches += 1   
			if iteration % 100 == 0:
				err, y_pred, acc = test_fn_rnn(inputs, targets)
				dis = [0] * NUM_CLASSES
				for _y in targets:
					dis[_y] += 1
				print (dis)
				print (y_pred)
				print("Batch " + str(iteration) +  ", Training Accuracy= " + \
								  "{0}".format(acc))
			iteration += 1

		y_pred, y_true = np.empty((0)), np.empty((0))
		test_err, test_batches = 0, 0
		for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False): 
			inputs, targets = batch 
			err, pred, acc = test_fn_rnn(inputs, targets) 
			test_err += err 
			y_true = np.append(y_true, targets, axis = 0) 
			y_pred = np.append(y_pred, pred.astype(np.uint8), axis = 0) 
			test_batches += 1  
		print("\t Recurrent: epoch {} of {} took {:.3f}s.".format(epoch + 1, num_epochs, time.time() - start_time)) 
		print("\t\ttraining loss:\t\t\t\t\t\t{:.6f}".format(train_err / train_batches)) 
		print("\t\ttest loss:\t\t\t\t\t\t\t{:.6f}".format(test_err / test_batches)) 
		print("\t\ttest fscore (weighted):\t\t\t\t{:.4f} ".format(f1_score(y_true, y_pred, average='weighted')* 100))  
		tr_l_rnn.append(train_err / train_batches ) 
		ts_l_rnn.append(test_err / train_batches )  
except KeyboardInterrupt: pass