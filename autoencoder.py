from tf_records import parse_ast_data

import tensorflow as tf
from tf_records import cond_tf_record_parser, COND_FEATURE_LENGTH as INPUT_UNITS


AST_INDEX = "ast"
CONDITION = "cond"

TRAIN_DATA_FILE = "Datasets/Hour of Code/tfrecords/train.tfrecords"
VAL_DATA_FILE = "Datasets/Hour of Code/tfrecords/val.tfrecords"
TEST_DATA_FILE = "Datasets/Hour of Code/tfrecords/test.tfrecords"

TRAIN_AST_FILE = "Datasets/Hour of Code/ast_data/train.ast"
VAL_AST_FILE = "Datasets/Hour of Code/ast_data/val.ast"
TEST_AST_FILE = "Datasets/Hour of Code/ast_data/test.ast"

H1_UNITS = 256
H2_UNITS = 128
LEARNING_RATE = 1e-2
REGULARIZER_COEF = 0.1

BATCH_SIZE = 64
NUM_EPOCHS = 60
SHUFFLE_BUFFER_SIZE = 100


train_iter = tf.data.TFRecordDataset(TRAIN_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .batch(BATCH_SIZE)\
    .make_initializable_iterator()
train_x, train_y, train_id = train_iter.get_next()

eval_iter = tf.data.TFRecordDataset(VAL_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .make_initializable_iterator()
eval_x, eval_y, eval_id = eval_iter.get_next()

test_iter = tf.data.TFRecordDataset(TEST_DATA_FILE)\
    .map(cond_tf_record_parser)\
    .make_initializable_iterator()
test_x, test_y, test_id = eval_iter.get_next()


train_ast_to_id, train_asts = parse_ast_data(TRAIN_AST_FILE)
eval_ast_to_id, eval_asts = parse_ast_data(VAL_AST_FILE)
test_ast_to_id, test_asts = parse_ast_data(TEST_AST_FILE)


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([INPUT_UNITS, H1_UNITS])),
    'decoder_h1': tf.Variable(tf.random_normal([H1_UNITS, INPUT_UNITS])),
}
program_matricies = []
for i in
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([H1_UNITS])),
    'decoder_b1': tf.Variable(tf.random_normal([INPUT_UNITS])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    return layer_1


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    return layer_1


def linear_map(x):
    transform = tf.matmul(x, weights['linear_map'])
    return transform



# Construct model
P = tf.placeholder(tf.float32, [None, INPUT_UNITS])
Q = tf.placeholder(tf.float32, [None, INPUT_UNITS])

autoencoder_op = decoder(encoder(P))
encoder_op = encoder(P)
linear_op = linear_map(encoder_op)
decoder_op = decoder(linear_op)

P_true, P_pred = P, autoencoder_op
Q_true, Q_pred = Q, decoder_op

auto_loss = tf.reduce_mean(tf.pow(P_true - P_pred, 2))
end_to_end_loss = tf.losses.sigmoid_cross_entropy(Q_true, Q_pred)
regularizer = tf.nn.l2_loss(weights['encoder_h1']) \
              + tf.nn.l2_loss(weights['decoder_h1']) \
              + tf.nn.l2_loss(weights['linear_map'])
loss = auto_loss + end_to_end_loss + REGULARIZER_COEF*regularizer

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

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
                _, train_loss_val = sess.run([optimizer, loss], feed_dict={
                    P: sess.run(train_x),
                    Q: sess.run(train_y)
                })
                print('Epoch %i: Minibatch Loss: %f' % (epoch, train_loss_val))
            except tf.errors.OutOfRangeError:
                break
        sess.run(eval_iter.initializer)
        eval_loss_val = sess.run([loss], feed_dict={
            P: sess.run(tf.expand_dims(eval_x, axis=0)),
            Q: sess.run(tf.expand_dims(eval_y, axis=0))
        })
        print('Epoch %i: Evaluation Loss: %f ####################################################' % (epoch, eval_loss_val[0]))

    # Evaluate on Test
    sess.run(eval_iter.initializer)
    test_loss_val = sess.run([loss], feed_dict={
        P: sess.run(tf.expand_dims(test_x, axis=0)),
        Q: sess.run(tf.expand_dims(test_y, axis=0))
    })
    print('Test Loss: %f $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' % (test_loss_val[0]))
