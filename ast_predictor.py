import tensorflow as tf

from matrix_predictor_tf_records import TF_RECORD_TRAIN, TF_RECORD_EVAL, TF_RECORD_TEST, tf_record_parser, H1_UNITS, MAX_SEQUENCE_LENGTH
from ast_tokenizer import NUM_TOKENS as TOKEN_DIMENSION


BATCH_SIZE = 64
MODEL_OUTPUT_DIM = TOKEN_DIMENSION
HIDDEN_STATE_SIZE = [4*MODEL_OUTPUT_DIM, 2*MODEL_OUTPUT_DIM, MODEL_OUTPUT_DIM]
NUM_EPOCHS = 100

data_iter_train = tf.data.TFRecordDataset(TF_RECORD_TRAIN)\
            .map(tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
sequences_train, matrices_train = data_iter_train.get_next()

data_iter_eval = tf.data.TFRecordDataset(TF_RECORD_EVAL)\
            .map(tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
sequences_eval, matrices_eval = data_iter_eval.get_next()

data_iter_test = tf.data.TFRecordDataset(TF_RECORD_TEST)\
            .map(tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
sequences_test, matrices_test = data_iter_test.get_next()


def multi_lstm_model():
    cells = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in HIDDEN_STATE_SIZE]
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
<<<<<<< HEAD
    
def matrix_replicates(matrices):
    '''matrices.shape == [num_matrices, H1_UNITS**2]
    returns: [num_matrices, MAX_SEQUENCE_LENGTH, H1_UNITS**2]
    '''
    num_matrices = matrices.shape[0]
    return tf.transpose(
        tf.reshape(
                tf.tile(matrices, [1, MAX_SEQUENCE_LENGTH]),
                [num_matrices,  H1_UNITS**2, MAX_SEQUENCE_LENGTH,]),
        [0, 2, 1])


Seqs = tf.placeholder(tf.float32, [None, MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION])
Mats = tf.placeholder(tf.float32, [None, MAX_SEQUENCE_LENGTH, H1_UNITS**2])

lstm_model = multi_lstm_model()
output, state = tf.nn.dynamic_rnn(lstm_model, matrix_replicates(Mats), dtype=tf.float32)
predicted_asts = output

loss = tf.losses.mean_squared_error(Seqs, predicted_asts)
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(NUM_EPOCHS):
        sess.run(data_iter_train.initializer)
        while True:
            try:
                # TODO: matrices_train is [BATCH_SIZE x H1**2] it needs to be [BATCH_SIZE x MAX_SEQUENCE_LENGTH x H1**2]
                #       the way we'll do this is by replicating 

                _, train_loss_val = sess.run([optimizer, loss], feed_dict={
                    Seqs: sess.run(sequences_train),
                    Mats: sess.run(matrices_train)
                })
                print('Epoch {}: Minibatch Loss: {}'.format(epoch, train_loss_val))
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                break  # typically happens when there isn't enough data for a given AST
        sess.run(data_iter_eval.initializer)
        eval_loss_val = sess.run([loss], feed_dict={
            Seqs: sess.run(sequences_eval),
            Mats: sess.run(matrices_eval)
        })
        print('Epoch %i: Evaluation Loss: %f ####################################################' % (
        epoch, eval_loss_val[0]))
test_loss_val = sess.run([loss], feed_dict={
    Seqs: sess.run(sequences_train),
    Mats: sess.run(matrices_test)
})
print('Test Loss: %f ####################################################' % test_loss_val[0])
