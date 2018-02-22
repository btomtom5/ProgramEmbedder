import tensorflow as tf

import numpy as np

import json
from os import listdir
from os.path import isfile, join

AST_INDEX = "ast"
CONDITION_KEY = "cond"
PRECONDITION_KEY = "precond"
POSTCONDITION_KEY = "postcond"


################################################################
# Defines the Tensorflow Dataset object for the autoencoder
################################################################

def data_set_from_file(file_path):
    """
    Assumes file_path is of the form some/path/<Number>.json
    """
    with open(file_path) as f:
        data = json.load(f)
    precondition_tensor = tf.convert_to_tensor(data[PRECONDITION_KEY], dtype=tf.float32)
    postcondition_tensor = tf.convert_to_tensor(data[POSTCONDITION_KEY], dtype=tf.float32)
    datapoint_1 = (precondition_tensor, precondition_tensor)
    datapoint_2 = (postcondition_tensor, postcondition_tensor)
    return [datapoint_1, datapoint_2]


def data_set_from_directory(dir_path):
    """
    Assumes that the only items in dir_path are files that
    have the name: <Number>.json
    """
    dataset = []
    for item_name in listdir(dir_path):
        full_path = join(dir_path, item_name)
        if isfile(full_path):
            dataset.extend(data_set_from_file(full_path))
    return dataset


def my_input_fn(data_dirs, perform_shuffle=False, repeat_count=0, buffer_size=256, batch_size=32):
    data = []
    for dir_path in data_dirs:
        data.extend(data_set_from_directory(dir_path))
    dataset = tf.data.Dataset.from_tensors(data)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator


################################################################
# Defines the Tensorflow Estimator object for the autoencoder
################################################################

# Program Arguments
DATA_PATH = [""]

# Hyper Parameters
NUM_INPUTS = 784
H1_UNITS = 256
H2_UNITS = 128
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 100


X = tf.placeholder("float", [None, NUM_INPUTS])
iterator = my_input_fn(DATA_PATH, True, NUM_EPOCHS, batch_size=BATCH_SIZE)
batch_x, batch_y = iterator.get_next()

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([NUM_INPUTS, H1_UNITS])),
    'encoder_h2': tf.Variable(tf.random_normal([H1_UNITS, H2_UNITS])),
    'decoder_h1': tf.Variable(tf.random_normal([H2_UNITS, H1_UNITS])),
    'decoder_h2': tf.Variable(tf.random_normal([H1_UNITS, NUM_INPUTS])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([H1_UNITS])),
    'encoder_b2': tf.Variable(tf.random_normal([H2_UNITS])),
    'decoder_b1': tf.Variable(tf.random_normal([H1_UNITS])),
    'decoder_b2': tf.Variable(tf.random_normal([NUM_INPUTS])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred, y_true = decoder_op, X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for epoch in range(NUM_EPOCHS):
        sess.run(iterator.initializer)
        while True:
            try:
                sess.run([optimizer], feed_dict={X: batch_x})
            except tf.errors.OutOfRangeError:
                break

        # Evaluate on Eval dataset
        l = sess.run([loss], feed_dict={X: batch_x}) # TODO: do this on an EVAL dataset
        print('Epoch %i: Minibatch Loss: %f' % (epoch, l))

    # Evaluate on Test
    # Encode and decode images from test set and visualize their reconstruction.
    # TODO: test on TEST dataset
