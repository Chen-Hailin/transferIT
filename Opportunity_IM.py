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
data_folder = "./data/"
result_path = os.path.join(os.getcwd(),"result")
## Parameters
# global_step = tf.Variable(0, trainable=False)
# base_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,
#                                           150, 0.9)
learning_rate = 0.0001
learning_rate_t = 0.00005
batch_size = 100
transfer_batch_size = 10
display_step = 15
# Network Parameters
SLIDING_WINDOW_LENGTH = 24  # timesteps
n_hidden = 128  # hidden layer num of features
n_hidden_d = 256
n_classes = 18  # Activity label, including null class
rnn_layers = 2  # Number of rnn layers
conv_multiplier = 64
# Session Parameters
SLIDING_WINDOW_STEP = 12
# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

'''
	funtions for get raw data
'''


def load_dataset(filename):
    f = open(filename, 'rb')
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


def load_dataset_s(i, j):
    data = np.load(data_path + 'S' + str(i) + '-ADL' + str(j) + '.dat_data.npy')
    labels = np.load(data_path + 'S' + str(i) + '-ADL' + str(j) + '.dat_activity_labels.npy')
    return (data, labels)


def load_all(i):
    data = np.load(data_path + 'S' + str(i) + '-Drill.dat_data.npy')
    labels = np.load(data_path + 'S' + str(i) + '-Drill.dat_activity_labels.npy')
    for j in [1, 2, 3]:
        data = np.vstack((data, np.load(data_path + 'S' + str(i) + '-ADL' + str(j) + '.dat_data.npy')))
        labels = np.vstack((labels, np.load(data_path + 'S' + str(i) + '-ADL' + str(j) + '.dat_activity_labels.npy')))
    return (data, labels)





def load_test(i):
    data = np.load(data_path + 'S' + str(i) + '-ADL4.dat_data.npy')
    labels = np.load(data_path + 'S' + str(i) + '-ADL4.dat_activity_labels.npy')
    data = np.vstack((data, np.load(data_path + 'S' + str(i) + '-ADL5.dat_data.npy')))
    labels = np.vstack((labels, np.load(data_path + 'S' + str(i) + '-ADL5.dat_activity_labels.npy')))
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
        segments = np.vstack([segments, np.array([data[start: end]])])
        labels = np.vstack((labels, activity_labels[end]))
    return segments, labels



def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def ReshapeActivityLabels(labels):
    # These are label unique indexes
    new_labels = np.zeros(shape=(labels.shape[0], 18))
    for i in range(labels.shape[0]):
        new_labels[i][labels[i]] = 1
    return new_labels


def trim_null(X, y):
    assert X.shape[0] == y.shape[0]
    nullset = []
    for i in range(y.shape[0]):
        if (y[i] == 0):
            nullset.append(i)
    X_trim = np.delete(X, nullset, axis=0)
    y_trim = np.delete(y, nullset, axis=0)
    return (X_trim, y_trim)

source_sub = 3
target_sub = 4


def load_with_cache():
    source_path = data_folder+"subject"+str(source_sub)
    target_path = data_folder+"subject"+str(target_sub)
    if(os.path.exists(source_path+"_X_train.npy")):
            X_train = np.load(source_path+'_X_train.npy')
            y_train = np.load(source_path+'_y_train.npy')
    else:
            raw_X_train, raw_train_activity_labels = load_all(source_sub)
            X_train, y_train = segment_signal(raw_X_train, raw_train_activity_labels)
            X_train = np.expand_dims(X_train, 1)
            np.save(source_path+"_X_train", X_train)
            np.save(source_path+"_y_train", y_train)
    return X_train, y_train



def load_mixed(j):
    source_path = data_folder + "subject" + str(source_sub)
    if (os.path.exists(source_path + "_X_train_mixed.npy")):
        x_train = np.load(source_path + '_X_train_mixed.npy')
        y_train = np.load(source_path + '_y_train_mixed.npy')
    else:
        raw_x_train, raw_train_activity_labels = load_all(source_sub)
        x_train, y_train = segment_signal(raw_x_train, raw_train_activity_labels)
    r_x_train_t, r_train_activity_labels_t = load_dataset_s(target_sub, j)
    x_target_train, y_target_train = segment_signal(r_x_train_t, r_train_activity_labels_t)
    x_train = np.vstack([x_train, x_target_train])
    y_train = np.vstack([y_train, y_target_train])
    assert x_train.shape[0] == y_train.shape[0]
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm,:,:]
    y_train = y_train[perm]
    x_train = np.expand_dims(x_train,1)
    return x_train, y_train


# def load_single_target(i,j):
#     X_train = np.load()

def load_twenty_target():
    target_path = data_folder + "subject" + str(target_sub)
    if (os.path.exists(target_path+"_labelled_data_20percent.npy") and os.path.exists(target_path+"_labelled_labels_20percent.npy")):
        x_train = np.load(target_path+"_labelled_data_20percent.npy")
        y_train = np.load(target_path+"_labelled_labels_20percent.npy")
        return x_train, y_train




'''
    Load data here
'''


#change here
x_train, y_train = load_with_cache()
# x_train_t, y_train_t = load_dataset_s(target_sub, 2)
# raw_x_train, raw_y_train = load_dataset_s(source_sub, 2)
# x_train, y_train = segment_signal(raw_x_train, raw_y_train)
# x_train = np.expand_dims(x_train, 1)
#change here
raw_X_test, raw_y_test = load_test(source_sub)


raw_X_test_t, raw_y_test_t = load_test(target_sub)

raw_x_train_t, raw_train_activity_labels_t = load_dataset_s(target_sub, 2)
x_train_t, y_train_t = segment_signal(raw_x_train_t, raw_train_activity_labels_t)
x_train_t = np.expand_dims(x_train_t, 1)

x_test, y_test = segment_signal(raw_X_test, raw_y_test)
x_test = np.expand_dims(x_test, 1)

x_test_t, y_test_t = segment_signal(raw_X_test_t, raw_y_test_t)
x_test_t = np.expand_dims(x_test_t, 1)

# x_train_t, y_train_t = load_single_target()

# raw_X_train, raw_train_activity_labels = load_all(source_sub)
#raw_X_train, raw_train_activity_labels = load_all(source_sub)


print("finish fetching")

# x_train = np.expand_dims(x_train, 1)

'''
	our model starts
'''

'''
	tensorflow wrapper functions for constructing CNN
'''


def weight_variable(shape):
    # initial = tf.orthogonal_initializer(shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def conv2d_t(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def apply_depthwise_conv(x, kernel_size, in_channels, mul_channels):
    weights = weight_variable([1, kernel_size, in_channels, mul_channels])
    biases = bias_variable([in_channels * mul_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))


def apply_conv(x, kernel_size, in_channels, out_channels):
    weights = weight_variable([1, kernel_size, in_channels, out_channels])
    biases = bias_variable([out_channels])
    return tf.nn.relu(tf.add(conv2d(x, weights), biases))


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

def apply_lstm_t(x, weights, biases, rnn_layers, keep_prob):
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
    # x = tf.squeeze(x)
    x = tf.reshape(x, [-1, input_size])
    weights = weight_variable([input_size, out_channels])
    biases = bias_variable([out_channels])
    return tf.nn.relu(tf.add(tf.matmul(x, weights), biases))


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

# def channel_seperate(x):

# last_weights = weight_variable([n_hidden, n_classes])
# last_biases = weight_variable([n_classes])

graph2 = tf.Graph()
with graph2.as_default():

    x_t = tf.placeholder("float", [None, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS])
    y_t = tf.placeholder("float", [None, n_classes])
    loss_weight = tf.placeholder("float", [batch_size])
    keep_prob_t = tf.placeholder("float")
    # _x_ = tf.variable(tf.empty())
    x_t_ = tf.transpose(x_t, [0, 3, 2, 1])
    # x_ shape [batch_size, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH, 1]
    # x_ = tf.split(x_, NB_SENSOR_CHANNELS)
    # for



    """
        Domain Classifier
    """

    with tf.variable_scope("d_c"):
        weight01 = tf.get_variable("weight01", shape=[1, 5, 1, conv_multiplier], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase01 = tf.get_variable("biase01", shape=[conv_multiplier], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        layer_cnn_1 = tf.nn.relu(tf.add(conv2d(x_t_, weight01), biase01))
        weight12 = tf.get_variable("weight12", shape=[1, 5, conv_multiplier, conv_multiplier], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase12 = tf.get_variable("biase12", shape=[conv_multiplier], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        layer_cnn_2 = tf.nn.relu(tf.add(conv2d(layer_cnn_1, weight12), biase12))
        weight_d1 =  tf.get_variable("weightd1", shape=[NB_SENSOR_CHANNELS*conv_multiplier*(SLIDING_WINDOW_LENGTH - 2 * 4), n_hidden_d], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase_d1 = tf.get_variable("biased1", shape=[n_hidden_d], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        layer_cnn_out = tf.reshape(layer_cnn_2,
                                   [-1, NB_SENSOR_CHANNELS * conv_multiplier * (SLIDING_WINDOW_LENGTH - 2 * 4)])
        layer_fc_1 = tf.add(tf.matmul(layer_cnn_out, weight_d1), biase_d1)
        weight_d2 = tf.get_variable("weightd2", shape=[n_hidden_d, 2], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase_d2 = tf.get_variable("biased2", shape=[2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0))
        layer_fc_2 = tf.add(tf.matmul(layer_fc_1, weight_d2), biase_d2)

        tf.get_variable_scope().reuse_variables()

    logits = layer_fc_2
    # d_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)) +\
    #     l2_beta * tf.nn.l2_loss(weight_d2) + l2_beta * tf.nn.l2_loss(weight_d1)
    # domain_train_op = tf.train.AdamOptimizer(learning_rate).minimize(d_loss)
    domain_predictions = tf.nn.softmax(logits)
    # correct_domain_prediction = tf.equal(tf.argmax(domain_predictions, 1), tf.argmax(y, 1))
    # accuracy_t = tf.reduce_mean(tf.cast(correct_domain_prediction, "float"))

    with tf.variable_scope("origin"):
        weights12_t = tf.get_variable("weight12tim", shape=[1, 5, 1, conv_multiplier])
        biases12_t = tf.get_variable("biase12tim", shape=[conv_multiplier])
        weights23_t = tf.get_variable("weight23tim", shape=[1, 5, conv_multiplier, conv_multiplier])
        biases23_t = tf.get_variable("biase23tim", shape=[conv_multiplier])
        weights34_t = tf.get_variable("weight34tim", shape=[1, 5, conv_multiplier, conv_multiplier])
        biases34_t = tf.get_variable("biase34tim", shape=[conv_multiplier])
        weights45_t = tf.get_variable("weight45tim", shape=[1, 5, conv_multiplier, conv_multiplier])
        biases45_t = tf.get_variable("biase45tim", shape=[conv_multiplier])
        last_weights_t = tf.get_variable("weightltim", shape=[n_hidden, n_classes], dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        last_biases_t = tf.get_variable("biaseltim", shape=[n_classes], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0))

    # layer2 = apply_conv(x_, 5, 1, conv_multiplier)
    layer2_t = tf.nn.relu(tf.add(tf.nn.conv2d(x_t_, weights12_t, strides=[1, 1, 1, 1], padding='VALID'), biases12_t))

    # layer_stop_grad_1 = tf.stop_gradient(layer2_t)
    # layer3 = apply_conv(layer2, 5, conv_multiplier, conv_multiplier)
    layer3_t = tf.nn.relu(tf.add(tf.nn.conv2d(layer2_t, weights23_t, strides=[1, 1, 1, 1], padding='VALID'), biases23_t))
    # layer4 = apply_conv(layer3, 5, conv_multiplier, conv_multiplier)
    # layer_stop_grad_2 = tf.stop_gradient(layer3_t)

    layer4_t = tf.nn.relu(tf.add(tf.nn.conv2d(layer3_t, weights34_t, strides=[1, 1, 1, 1], padding='VALID'), biases34_t))
    # layer5 = apply_conv(layer4, 5, conv_multiplier, conv_multiplier)

    layer5_t = tf.nn.relu(tf.add(tf.nn.conv2d(layer4_t, weights45_t, strides=[1, 1, 1, 1], padding='VALID'), biases45_t))

    layer67_t = apply_lstm_t(layer5_t, last_weights_t, last_biases_t, rnn_layers, keep_prob_t)
    layerOut_t = layer67_t
    result_tranfer = tf.nn.softmax(layerOut_t)

    #


    #

    # cross_entropy_transfer = -tf.reduce_sum(y_t * tf.log(result_tranfer))
    #
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_t, logits=layerOut_t, weights=loss_weight))
    #
    all_vars_t = tf.trainable_variables()
    lossL2_t = tf.add_n([tf.nn.l2_loss(v) for v in all_vars_t if 'bias' not in v.name and "tim" in v.name]) * 0.0005

    loss_t = tf.add(loss, lossL2_t)

    global_step_t = tf.Variable(0, trainable=False)
    train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_t, global_step=global_step_t)
    correct_prediction_transfer = tf.equal(tf.argmax(result_tranfer, 1), tf.argmax(y_t, 1))
    accuracy_transfer = tf.reduce_mean(tf.cast(correct_prediction_transfer, "float"))

    gpu_options_t = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    saver_t = tf.train.Saver({"weight01": weight01, "weight12": weight12, "weightd1": weight_d1, "weightd2": weight_d2,
         "biase01": biase01, "biase12": biase12, "biased1": biase_d1, "biased2": biase_d2})
    # saver = tf.train.Saver({"weight12": weights12, "weight23": weights23, "weight34": weights34, "weight45": weights45, "weightl": last_weights, "biase12": biases12, "biase23": biases23, "biase34": biases34, "biase45": biases45, "biasel": last_biases})


    # Initializing the variables
    init_t = tf.global_variables_initializer()
    #init_transfer_op = tf.variables_initializer([last_weights, last_biases])



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

    with tf.Session(graph=graph2, config=tf.ConfigProto(gpu_options=gpu_options_t)) as session_transfer:

        session_transfer.run(init_t)

        #change here
        saver_t.restore(session_transfer, "/tmp/model_d.ckpt")

        print("Session 2: Model restored")
        source_result=[]
        target_result=[]

        for iteration in range(101):
            # start_random = random.randint(0, batch_size)
            # start = start_random
            step = 0

            g_min = np.array([])
            g_max = np.array([])


            # for batch in iterate_minibatches(x_train, y_train, batch_size):
            #     pred = logits.eval(feed_dict={x_t: batch_x})
            #     mmin =

            for batch in iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
                batch_x, batch_y = batch

                raw_map_vector = logits.eval(feed_dict={x_t:batch_x})
                m_min = np.min(raw_map_vector,0)
                raw_map_vector = (raw_map_vector - m_min)
                m_max = np.max(raw_map_vector,0)
                raw_map_vector = raw_map_vector / m_max

                map_vector_s = raw_map_vector[:,0]
                map_vector_t = raw_map_vector[:, 1]
                map_vector_s = np.exp(map_vector_s)

                map_vector = map_vector_t / map_vector_s
                m_min = np.min(map_vector)
                map_vector = (map_vector - m_min)
                m_max = np.max(map_vector)
                map_vector = map_vector/ m_max

                map_vector = np.exp(np.multiply(map_vector,5))/20

                # session_transfer.run(train_op, feed_dict={x_t: batch_x, y_t: batch_y, keep_prob_t: 0.5})
                acc = session_transfer.run(accuracy_transfer, feed_dict={x_t: batch_x, y_t: batch_y, loss_weight: map_vector, keep_prob_t: 1.0})
                if (acc < 0.7 and iteration > 20):
                    session_transfer.run(train_op, feed_dict={x_t: batch_x, y_t: batch_y, loss_weight: map_vector, keep_prob_t: 0.5})
                if (acc < 0.8 + 0.0015 * iteration):
                    session_transfer.run(train_op, feed_dict={x_t: batch_x, y_t: batch_y, loss_weight: map_vector, keep_prob_t: 0.5})

            # Classification of the testing data
            # print("Testing: Processing {0} instances in mini-batches of {1}".format(X_test.shape[0], batch_size))
            test_pred = np.empty((0))
            test_true = np.empty((0))
            # tf_confusion_metrics(result, y_train[start:], sess, {x : X_train[start:]})
            for batch in iterate_minibatches(x_test, y_test, batch_size):
                inputs, targets = batch
                acc, y_pred = session_transfer.run([accuracy_transfer, tf.argmax(result_tranfer, 1)], feed_dict={x_t: inputs, y_t: targets, keep_prob_t: 1.0})
                test_pred = np.append(test_pred, y_pred, axis=0)
                test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)
            f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
            source_result.append(f1)
            print("Iter " + str(iteration) + " Subject" + str(source_sub) + " Test F1 score= " + "{:.5f}".format(f1))


            test_pred = np.empty((0))
            test_true = np.empty((0))
            # tf_confusion_metrics(result, y_train[start:], sess, {x : X_train[start:]})
            for batch in iterate_minibatches(x_test_t, y_test_t, batch_size):
                inputs, targets = batch
                acc, y_pred = session_transfer.run([accuracy_transfer, tf.argmax(result_tranfer, 1)],
                                                   feed_dict={x_t: inputs, y_t: targets, keep_prob_t: 1.0})
                test_pred = np.append(test_pred, y_pred, axis=0)
                test_true = np.append(test_true, np.argmax(targets, axis=1), axis=0)
            f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
            print("Iter " + str(iteration) + " Subject" + str(target_sub) + " Test F1 score= " + "{:.5f}".format(f1))
            target_result.append(f1)
            if (f1 > 0.39):
                f1 = sk.metrics.f1_score(test_true, test_pred, average=None)
                print("Iter " + str(iteration) + " Subject" + str(target_sub) + " Test F1 score= ")
                print(f1)

        filename = os.path.join(result_path, "im" + str(source_sub) + str(target_sub) + ".csv")
        with open(filename, "w") as fout:
            fout.write("Iteration,source f1 score,target f1 score\n")
            for i in range(len(source_result)):
                fout.write(str(i + 1) + "," + str(source_result[i]) + "," + str(target_result[i]) + "\n")





