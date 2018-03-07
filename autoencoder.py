import tensorflow as tf
from tf_records_writer import cond_tf_record_parser


AST_INDEX = "ast"
CONDITION = "cond"

TRAIN_DATA_FILE = "Datasets/Hour of Code/tfrecords/train.tfrecords"
VAL_DATA_FILE = "Datasets/Hour of Code/tfrecords/val.tfrecords"
TEST_DATA_FILE = "Datasets/Hour of Code/tfrecords/test.tfrecords"

INPUT_UNITS = 784
H1_UNITS = 256
H2_UNITS = 128
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 100
SHUFFLE_BUFFER_SIZE = 100


X = tf.placeholder("float", [None, INPUT_UNITS])
iterator = tf.data.TFRecordDataset(TEST_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .batch(BATCH_SIZE)\
    .make_initializable_iterator()
batch_x = iterator.get_next()


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([INPUT_UNITS, H1_UNITS])),
    'encoder_h2': tf.Variable(tf.random_normal([H1_UNITS, H2_UNITS])),
    'decoder_h1': tf.Variable(tf.random_normal([H2_UNITS, H1_UNITS])),
    'decoder_h2': tf.Variable(tf.random_normal([H1_UNITS, INPUT_UNITS])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([H1_UNITS])),
    'encoder_b2': tf.Variable(tf.random_normal([H2_UNITS])),
    'decoder_b1': tf.Variable(tf.random_normal([H1_UNITS])),
    'decoder_b2': tf.Variable(tf.random_normal([INPUT_UNITS])),
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
