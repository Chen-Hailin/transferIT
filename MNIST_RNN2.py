from __future__ import print_function
import os
import numpy as np
import math
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.contrib import rnn
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

## Parameters
learning_rate = 0.001
batch_size = 128
display_step = 10
training_iters = 100000
# Network Parameters
n_input = 28 # opportunity data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # Activity label, including null class
rnn_layers = 1 # Number of rnn layers
# Session Parameters
sliding_step_size = 13


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
'''
def fetch(file_path, size):
    this_batch = np.empty([0, size])
    this_batch = np.load(data_path + file_path)
    return this_batch
data_path = os.getcwd() + "/output1/"

# Import Opportunity data
data = np.empty([0, n_input])
#data = np.load(data_path + 'S4-Drill.dat_data.npy')
activity_labels = np.empty([0, n_classes])
#activity_labels = np.load(data_path + 'S4-Drill.dat_activity_labels.npy')

## get all the data for person1, ADL 1-3 plus drill run, as training data;
## get ADL 4 as testing data
i = 1
for j in range (1, 4):
    data = np.vstack((data, fetch(('S'+str(i)+'-ADL'+str(j)+'.dat_data.npy'), n_input) ))
    activity_labels = np.vstack((activity_labels, fetch(('S'+str(i)+'-ADL'+str(j)+'.dat_activity_labels.npy'), n_input) ))
data = np.vstack((data, fetch(('S'+str(i)+'-Drill.dat_data.npy'), n_input) ))
activity_labels = np.vstack((activity_labels, fetch(('S'+str(i)+'-Drill.dat_activity_labels.npy'), n_classes) ))
assert (data.shape[0] == activity_labels.shape[0]) , "wrong ! "
print ("total rows of training data: "+str(data.shape[0]))

test_data = np.load(data_path + 'S1-ADL4.dat_data.npy')
test_activity_labels = np.load(data_path + 'S1-ADL4.dat_activity_labels.npy')
'''


#second parameter, dependent on training data shape
#total_time_steps = data.shape[0]
#max_batch_steps = (total_time_steps - (n_steps - sliding_step_size) ) // (sliding_step_size * batch_size)
#max_batch_steps = 700

# tf Graph inputs
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    #lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          n_hidden, forget_bias=1.0, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(rnn_layers)], state_is_tuple=True)
    #cell_simple = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    initial_state = cell.zero_state(batch_size, tf.float32)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32, initial_state = initial_state)

    # Linear activation, using rnn inner loop last output
    return_value = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return return_value


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.InteractiveSession()
sess.run(init)
batch_step = 0
# Keep training until reach max iterations
while batch_step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    '''
    batch_x = np.empty([0, n_steps, n_input])
    batch_y = np.empty([0, n_classes])
    start = batch_step * batch_size * sliding_step_size
    for i in range(batch_size):
        batch_x = np.vstack((batch_x, np.array([data[start : start + n_steps]])))
        batch_y = np.vstack((batch_y, activity_labels[start + n_steps - 1]))
        start += sliding_step_size
    '''
    #now batch_x.shape = [batch_size, n_steps, n_input]
    #    batch_y.shpae = [batch_size, n_classes]
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    if batch_step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        preds = sess.run(tf.argmax(pred, 1), feed_dict={x: batch_x, y: batch_y})
        '''
        #for debugging
        if (acc == 0 or acc > 0.3 or 1):
            dis = [0] * n_classes
            y_ = np.argmax(batch_y, axis = 1)
            for _y in y_:
                dis[_y] += 1
            print (dis)
            print (preds)
        '''
        #
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(batch_step*batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
    batch_step += 1
print("Optimization Finished!")

test_len = 128
#test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
#test_label = mnist.test.labels[:test_len]

for batch_step in range(2):
    batch_x = mnist.test.images[batch_step * test_len : (batch_step + 1) * test_len].reshape((-1, n_steps, n_input))
    batch_y = mnist.test.labels[batch_step * test_len : (batch_step + 1) * test_len]
    '''
    batch_x = np.empty([0, n_steps, n_input])
    batch_y = np.empty([0, n_classes])
    start = batch_step * batch_size * sliding_step_size
    for i in range(100):
        batch_x = np.vstack((batch_x, np.array([test_data[start : start + n_steps]])))
        batch_y = np.vstack((batch_y, test_activity_labels[start + n_steps - 1]))
        start += sliding_step_size
    '''
    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
#for debugging:
    preds = sess.run(tf.argmax(pred, 1), feed_dict={x: batch_x, y: batch_y})
    if (acc == 0 or acc > 0.3 or 1):
        dis = [0] * n_classes
        y_ = np.argmax(batch_y, axis = 1)
        for _y in y_:
            dis[_y] += 1
        print (dis)
        print (preds)
#    
    print("Testing Accuracy:", acc)
