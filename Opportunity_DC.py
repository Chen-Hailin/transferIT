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
SOURCE_PATH = "/home/hailin/transfer" #change here
DATA_FOLDER = "data"
data_path = os.getcwd() + "/../output1/"
data_folder = "./data/"
## Parameters
# global_step = tf.Variable(0, trainable=False)
# base_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(base_learning_rate, global_step,
#                                           150, 0.9)
learning_rate = 0.01
batch_size = 100
transfer_batch_size = 10
display_step = 15
# Network Parameters
SLIDING_WINDOW_LENGTH = 24  # timesteps
n_hidden = 256  # hidden layer num of features
n_classes = 18  # Activity label, including null class
rnn_layers = 2  # Number of rnn layers
conv_multiplier = 64
# Session Parameters
SLIDING_WINDOW_STEP = 12
# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
l2_beta = 0.001

source_sub = 3
target_sub = 4

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


def load_all(i):
    data = np.load(data_path + 'S' + str(i) + '-Drill.dat_data.npy')
    labels = np.load(data_path + 'S' + str(i) + '-Drill.dat_activity_labels.npy')
    for j in [1, 2, 3]:
        data = np.vstack((data, np.load(data_path + 'S' + str(i) + '-ADL' + str(j) + '.dat_data.npy')))
        labels = np.vstack((labels, np.load(data_path + 'S' + str(i) + '-ADL' + str(j) + '.dat_activity_labels.npy')))
    return (data, labels)


def load_with_cache(i):
    source_path = data_folder+"subject"+str(i)
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



def load_unlabelled_data_domain(s,t):
    file_dir = os.path.join(SOURCE_PATH, DATA_FOLDER)
    file_name = os.path.join(file_dir, "subject" + str(s) + "_unlabelled_data.npy")
    file_name_target = os.path.join(file_dir, "subject" + str(t) + "_unlabelled_data.npy")
    if (os.path.exists(file_name)):
        x_train = np.load(file_name)
    else:
        x_train, _ = load_with_cache(s)
    if (os.path.exists(file_name_target)):
        x_train_target = np.load(file_name_target)
    else:
        x_train_target, _ = load_with_cache(t)
    x_train_target = np.squeeze(x_train_target)
    x_train = np.squeeze(x_train)
    y_train = np.full([x_train.shape[0],2],[1,0],dtype=float)
    y_train_target = np.full([x_train_target.shape[0],2],[0,1],dtype=float)

    return x_train, y_train, x_train_target, y_train_target


def load_processed_domain_test(i):
    data = np.load(data_path + 'S' + str(i) + '-ADL4.dat_data.npy')
    data = np.vstack((data, np.load(data_path + 'S' + str(i) + '-ADL5.dat_data.npy')))
    labels = np.load(data_path + 'S' + str(i) + '-ADL4.dat_activity_labels.npy')
    labels = np.vstack((labels, np.load(data_path + 'S' + str(i) + '-ADL5.dat_activity_labels.npy')))
    data, _ = segment_signal(data, labels)
    if (i == target_sub):
        new_label = [0,1]
    else:
        new_label = [1,0]
    labels = np.full((data.shape[0],2),new_label,dtype=float)
    return data, labels

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

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


def accuracy(predictions, labels):
    return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

x_train, y_train, x_target_train, y_target_train = load_unlabelled_data_domain(3,4)

x_train = np.vstack([x_train,x_target_train])
y_train = np.vstack([y_train, y_target_train])
perm = np.random.permutation(x_train.shape[0])
assert x_train.shape[0] == y_train.shape[0]
x_train = x_train[perm, :, :]
y_train = y_train[perm, :]
x_train = np.expand_dims(x_train, 1)

x_test, y_test = load_processed_domain_test(3)
x_test_t, y_test_t = load_processed_domain_test(4)
x_test = np.vstack([x_test,x_test_t])
y_test = np.vstack([y_test,y_test_t])
assert x_test.shape[0] == y_test.shape[0]
x_test = np.expand_dims(x_test, 1)


graph_domain = tf.Graph()


with graph_domain.as_default():
    x = tf.placeholder("float", [None, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS])
    y = tf.placeholder("float", [None, 2])
    # keep_prob_t = tf.placeholder("float")
    # _x_ = tf.variable(tf.empty())
    x_ = tf.transpose(x, [0, 3, 2, 1])


    with tf.variable_scope("d_c"):
        weight01 = tf.get_variable("weight01", shape=[1, 5, 1, conv_multiplier], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase01 = tf.get_variable("biase01", shape=[conv_multiplier], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        layer_cnn_1 = tf.nn.relu(tf.add(conv2d(x_, weight01), biase01))
        weight12 = tf.get_variable("weight12", shape=[1, 5, conv_multiplier, conv_multiplier], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase12 = tf.get_variable("biase12", shape=[conv_multiplier], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        layer_cnn_2 = tf.nn.relu(tf.add(conv2d(layer_cnn_1, weight12), biase12))
        weight_d1 =  tf.get_variable("weightd1", shape=[NB_SENSOR_CHANNELS*conv_multiplier*(SLIDING_WINDOW_LENGTH - 2 * 4), n_hidden], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase_d1 = tf.get_variable("biased1", shape=[n_hidden], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0))
        layer_cnn_out = tf.reshape(layer_cnn_2,
                                   [-1, NB_SENSOR_CHANNELS * conv_multiplier * (SLIDING_WINDOW_LENGTH - 2 * 4)])
        layer_fc_1 = tf.add(tf.matmul(layer_cnn_out, weight_d1), biase_d1)
        weight_d2 = tf.get_variable("weightd2", shape=[n_hidden, 2], dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        biase_d2 = tf.get_variable("biased2", shape=[2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0))
        layer_fc_2 = tf.add(tf.matmul(layer_fc_1, weight_d2), biase_d2)

        tf.get_variable_scope().reuse_variables()

    logits = layer_fc_2
    d_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)) +\
        l2_beta * tf.nn.l2_loss(weight_d2) + l2_beta * tf.nn.l2_loss(weight_d1)
    domain_train_op = tf.train.AdamOptimizer(learning_rate).minimize(d_loss)
    domain_predictions = tf.nn.softmax(logits)
    correct_domain_prediction = tf.equal(tf.argmax(domain_predictions, 1), tf.argmax(y, 1))
    # accuracy_t = tf.reduce_mean(tf.cast(correct_domain_prediction, "float"))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver({"weight01": weight01, "weight12": weight12, "weightd1": weight_d1, "weightd2": weight_d2,
         "biase01": biase01, "biase12": biase12, "biased1": biase_d1, "biased2": biase_d2})


def main():
    with tf.Session(graph=graph_domain, config=tf.ConfigProto(gpu_options=gpu_options)) as d_session:
        d_session.run(init)

        for iteration in range(1):
            count = 0
            for batch_x, batch_y in iterate_minibatches(x_train, y_train, batch_size, shuffle=True):
                d_session.run([domain_train_op], feed_dict={x:batch_x, y:batch_y})
                count += 1
                if (count == 100):
                    break;

                # acc = d_session.run(accuracy_t, feed_dict={x:batch_x, y:batch_y})
                # acc = accuracy(pred, batch_y)
                # acc = d_session.run(accuracy_t, feed_dict={x:batch_x, y:batch_y})
                # print("Train accuracy:" + str(acc))
            acc_m = np.empty(0)
            test_pred = np.empty((0))
            test_true = np.empty((0))
            for batch_x, batch_y in iterate_minibatches(x_test, y_test, batch_size):
                pred = d_session.run(tf.argmax(domain_predictions, 1), feed_dict={x: batch_x, y: batch_y})
                acc = accuracy(domain_predictions.eval(feed_dict={x: batch_x}), batch_y)
                np.append(acc_m, acc)
                test_pred = np.append(test_pred, pred, axis=0)
                test_true = np.append(test_true, np.argmax(batch_y, axis=1), axis=0)
                logitss = logits.eval(feed_dict={x: batch_x})
                print(logitss)

            f1 = sk.metrics.f1_score(test_true, test_pred, average="weighted")
            print("Iter " + str(iteration) + " Subject" + str(target_sub) + " Test F1 score= " + "{:.5f}".format(f1))
            f1 = sk.metrics.f1_score(test_true, test_pred, average=None)
            print("Iter " + str(iteration) + " Subject" + str(target_sub) + " Test F1 score= ")
            print(f1)
            save_path = saver.save(d_session, "/tmp/model_d.ckpt")
            print("saved in: " + save_path)
            # acc = np.mean(acc_m)
            # print("Iteration " + str(iteration) + " accuracy: " + str(acc))

















