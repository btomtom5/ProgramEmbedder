import tensorflow as tf
import sys

from matrix_predictor_tf_records import TF_RECORD_TRAIN, TF_RECORD_EVAL, TF_RECORD_TEST, tf_record_parser, H1_UNITS, MAX_SEQUENCE_LENGTH
from ast_tokenizer import NUM_TOKENS as TOKEN_DIMENSION

# <<<<<<<<<<<<<<<<<<<<<<<<<< TODO: CHANGE EPOCHS + BATCH SIZE BACK TO AN APPROPRIATE AMOUNT >>>>>>>>>>>>>>
DATA_DIR = None
BATCH_SIZE = None
MODEL_OUTPUT_DIM = H1_UNITS**2
HIDDEN_STATE_SIZE = [4*MODEL_OUTPUT_DIM, 2*MODEL_OUTPUT_DIM, MODEL_OUTPUT_DIM]
NUM_EPOCHS = None


training_loss_log = []
MATRIX_PREDICTOR_TRAIN_LOGS = "Datasets/hour_of_code/results/matrix_predictor_train_log.txt"
evaluation_loss_log = []
MATRIX_PREDICTOR_EVAL_LOGS = "Datasets/hour_of_code/results/matrix_predictor_eval_log.txt"
test_loss_log = []
MATRIX_PREDICTOR_TEST_LOGS = "Datasets/hour_of_code/results/matrix_predictor_test_log.txt"

if __name__ == "__main__":
    if len(sys.argv) > 2:
        DATA_DIR = sys.argv[1]
        NUM_EPOCHS = int(sys.argv[2])
        BATCH_SIZE = int(sys.argv[3])
    else:
        raise Exception("Usage Error: python3 script_name.py <data | dev_data>")


data_iter_train = tf.data.TFRecordDataset(TF_RECORD_TRAIN % DATA_DIR)\
            .map(tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
sequences_train, matrices_train = data_iter_train.get_next()

data_iter_eval = tf.data.TFRecordDataset(TF_RECORD_EVAL % DATA_DIR)\
            .map(tf_record_parser)\
            .make_initializable_iterator()

sequences_eval, matrices_eval = data_iter_eval.get_next()

data_iter_test = tf.data.TFRecordDataset(TF_RECORD_TEST % DATA_DIR)\
            .map(tf_record_parser)\
            .make_initializable_iterator()
sequences_test, matrices_test = data_iter_test.get_next()


def multi_lstm_model():
    cells = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in HIDDEN_STATE_SIZE]
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


Seqs = tf.placeholder(tf.float32, [None, MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION])
Mats = tf.placeholder(tf.float32, [None, H1_UNITS**2])

lstm_model = multi_lstm_model()
output, state = tf.nn.dynamic_rnn(lstm_model, Seqs, dtype=tf.float32)
predicted_matrices = tf.unstack(tf.gather(output, [MAX_SEQUENCE_LENGTH - 1], axis=1), axis=1)[0]

loss = tf.losses.mean_squared_error(Mats, predicted_matrices)
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(NUM_EPOCHS):
        sess.run(data_iter_train.initializer)
        while True:
            try:
                _, train_loss_val = sess.run([optimizer, loss], feed_dict={
                    Seqs: sess.run(sequences_train),
                    Mats: sess.run(matrices_train)
                })
                training_loss_log.append((epoch, train_loss_val))
                print('Epoch {}: Minibatch Loss: {}'.format(epoch, train_loss_val))
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                break  # typically happens when there isn't enough data for a given AST
        sess.run(data_iter_eval.initializer)

        eval_loss_val = sess.run([loss], feed_dict={
            # evaluation set does not have batches, but placeholder requires batch size
            Seqs: sess.run(tf.expand_dims(sequences_eval, axis=0)),
            Mats: sess.run(tf.expand_dims(matrices_eval, axis=0))
        })
        evaluation_loss_log.append((epoch, eval_loss_val[0]))
        print('Epoch %i: Evaluation Loss: %f ####################################################' % (
        epoch, eval_loss_val[0]))
    sess.run(data_iter_test.initializer)

    test_loss_val = sess.run([loss], feed_dict={
        # test set does not have batches, but placeholder requires batch size
        Seqs: sess.run(tf.expand_dims(sequences_test, axis=0)),
        Mats: sess.run(tf.expand_dims(matrices_test, axis=0))
    })
    test_loss_log.append(test_loss_val[0])
    print('Test Loss: %f ####################################################' % test_loss_val[0])

with open(MATRIX_PREDICTOR_TRAIN_LOGS, 'w') as file:
    for record in training_loss_log:
        file.write(str(record))

with open(MATRIX_PREDICTOR_EVAL_LOGS, 'w') as file:
    for record in evaluation_loss_log:
        file.write(str(record))

with open(MATRIX_PREDICTOR_TEST_LOGS, 'w') as file:
    for record in test_loss_log:
        file.write(str(record))