import tensorflow as tf

from matrix_predictor_tf_records import TF_RECORD_FILE, tf_record_parser, H1_UNITS, MAX_SEQUENCE_LENGTH
from ast_tokenizer import NUM_TOKENS as TOKEN_DIMENSION


BATCH_SIZE = 64
HIDDEN_STATE_SIZE = [200, 100, H1_UNITS]
NUM_LSTM_LAYERS = 3
NUM_EPOCHS = 100

data_iter = tf.data.TFRecordDataset(TF_RECORD_FILE)\
            .map(tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
sequences, matrices = data_iter.get_next()


def multi_lstm_model():
    cells = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in HIDDEN_STATE_SIZE]
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


Seqs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION])
Mats = tf.placeholder(tf.float32, [BATCH_SIZE, H1_UNITS, H1_UNITS])

lstm_model = multi_lstm_model()
output, state = tf.nn.dynamic_rnn(lstm_model, Seqs)
predicted_matrices = tf.gather(output, [MAX_SEQUENCE_LENGTH - 1], axis=1)

loss = tf.losses.mean_squared_error(Mats, predicted_matrices)
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(NUM_EPOCHS):
        sess.run(data_iter.initializer)
        while True:
            try:
                _, train_loss_val = sess.run([optimizer, loss], feed_dict={
                    Seqs: sess.run(sequences),
                    Mats: sess.run(matrices)
                })
                print('File: {} Epoch {}: Minibatch Loss: {}'.format(file, epoch, train_loss_val))
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                break  # typically happens when there isn't enough data for a given AST
        # TODO: compute loss on eval dataset
# TODO: computer loss on test dataset
