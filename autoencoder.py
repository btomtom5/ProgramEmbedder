import tensorflow as tf
from tf_records_writer import cond_tf_record_parser, COND_FEATURE_LENGTH


AST_INDEX = "ast"
CONDITION = "cond"

TRAIN_DATA_FILE = "Datasets/Hour of Code/tfrecords/train.tfrecords"
VAL_DATA_FILE = "Datasets/Hour of Code/tfrecords/val.tfrecords"
TEST_DATA_FILE = "Datasets/Hour of Code/tfrecords/test.tfrecords"

INPUT_UNITS = COND_FEATURE_LENGTH
H1_UNITS = 256
H2_UNITS = 128
LEARNING_RATE = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 5
SHUFFLE_BUFFER_SIZE = 100


train_iter = tf.data.TFRecordDataset(TRAIN_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .batch(BATCH_SIZE)\
    .make_initializable_iterator()
train_x = train_iter.get_next()

eval_iter = tf.data.TFRecordDataset(VAL_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .make_initializable_iterator()
eval_x = eval_iter.get_next()

test_iter = tf.data.TFRecordDataset(TEST_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .make_initializable_iterator()
test_x = eval_iter.get_next()


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
X = tf.placeholder(tf.float32, [None, INPUT_UNITS])
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
        sess.run(train_iter.initializer)
        while True:
            try:
                _, train_loss_val = sess.run([optimizer, loss], feed_dict={X: sess.run(train_x)})
                print('Epoch %i: Minibatch Loss: %f' % (epoch, train_loss_val))
            except tf.errors.OutOfRangeError:
                break
        sess.run(eval_iter.initializer)
        eval_loss_val = sess.run([loss], feed_dict={X: sess.run(tf.expand_dims(eval_x, axis=0))})
        print('Epoch %i: Evaluation Loss: %f' % (epoch, eval_loss_val[0]))

    # Evaluate on Test
    sess.run(eval_iter.initializer)
    test_loss_val = sess.run([loss], feed_dict={X: sess.run(tf.expand_dims(test_x, axis=0))})
    print('Test Loss: %f' % (test_loss_val[0]))
