import tensorflow as tf

from matrix_learner_tf_records import COND_FEATURE_LENGTH
from matrix_predictor_tf_records import MAX_SEQUENCE_LENGTH
from ast_tokenizer import NUM_TOKENS as TOKEN_DIMENSION
from neural_predictor_tf_records import tf_record_parser, TF_RECORDS_TRAIN, TF_RECORDS_EVAL, TF_RECORDS_TEST


NUM_EPOCHS = 100
BATCH_SIZE = 32
LSTM_CELLS = [200, 100, 50]
PREDICTED_NN_LAYERS = [COND_FEATURE_LENGTH, 100, 50, 25, COND_FEATURE_LENGTH]


data_iter_train = tf.data.TFRecordDataset(TF_RECORDS_TRAIN)\
            .map(tf_record_parser)\
            .batch(BATCH_SIZE)\
            .make_initializable_iterator()
Ast_Seqs_train, Ps_train, Qs_train = data_iter_train.get_next()

data_iter_eval = tf.data.TFRecordDataset(TF_RECORDS_EVAL)\
            .map(tf_record_parser)\
            .make_initializable_iterator()
Ast_Seqs_eval, Ps_eval, Qs_eval = data_iter_train.get_next()

data_iter_test = tf.data.TFRecordDataset(TF_RECORDS_TEST)\
            .map(tf_record_parser)\
            .make_initializable_iterator()
Ast_Seqs_test, Ps_test, Qs_test = data_iter_train.get_next()


def multi_lstm_model():
    cells = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in LSTM_CELLS]
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


def post_lstm_fc_layer(lstm_output):
    weights = tf.Variable(tf.random_normal([LSTM_CELLS[-1], COND_FEATURE_LENGTH]), name='post_lstm_weights')
    biases = tf.Variable(tf.random_normal([LSTM_CELLS[-1]]), name='post_lstm_biases')
    return tf.nn.relu_layer(lstm_output, weights, biases, name='post_lstm_layer')


def reshape_concatenated_weights_to_layers(concatenated_weights, preconds):
    '''
    Takes in a hidden state (output from an LSTM model).
    Reshapes the hidden_state into a neural network with 3 hidden layers
    of size 3x, 2x, 1x.
    Hidden layers use ReLU.
    Assumes that PREDICTED_NN_LAYERS has at least 2 layer sizes

    :param hidden_state: the (concatenated) weights of the "predicted" feed-forward
    network that will represent functionality of the AST on the precond --> postcond
    :return: the outputs of the feed-forward network (sigmoid)
    '''
    start = 0
    layer = preconds
    for i in range(1, len(PREDICTED_NN_LAYERS) - 1):
        end = PREDICTED_NN_LAYERS[i-1]*PREDICTED_NN_LAYERS[i]
        weights = tf.reshape(
            tf.gather(concatenated_weights, list(range(start, end)), axis=0),
            [PREDICTED_NN_LAYERS[i-1], PREDICTED_NN_LAYERS[i]]
        )
        start = end
        end = start + PREDICTED_NN_LAYERS[0]
        biases = tf.gather(concatenated_weights, list(range(start, end)), axis=0)
        layer = tf.nn.relu_layer(layer, weights, biases)
    return tf.nn.sigmoid(layer)


# Define the model (in sequential order pretty much)
Ast_Seqs = tf.placeholder(tf.int32, [BATCH_SIZE, MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION])
lstm_model = multi_lstm_model()
all_outputs, _ = tf.nn.dynamic_rnn(lstm_model, Ast_Seqs, dtype=tf.float32)
final_output = tf.unstack(tf.gather(all_outputs, [MAX_SEQUENCE_LENGTH - 1], axis=1), axis=1)[0]
nn_as_vec = post_lstm_fc_layer(final_output)
Ps = tf.placeholder(tf.int32, [BATCH_SIZE, COND_FEATURE_LENGTH])
Qs_pred = reshape_concatenated_weights_to_layers(nn_as_vec, Ps)
Qs = tf.placeholder(tf.int32, [BATCH_SIZE, COND_FEATURE_LENGTH])

# TODO: define what accuracy means for Q_pred/Q
loss = tf.losses.mean_squared_error(Qs, Qs_pred)
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
                    Ast_Seqs: sess.run(Ast_Seqs_train),
                    Ps: sess.run(Ps_train),
                    Qs: sess.run(Qs_train),
                })
                print('Epoch {}: Minibatch Loss: {}'.format(epoch, train_loss_val))
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                break  # typically happens when there isn't enough data for a given AST
        sess.run(data_iter_eval.initializer)
        eval_loss_val = sess.run([loss], feed_dict={
            # evaluation set does not have batches, but placeholder requires batch size
            Ast_Seqs: sess.run(tf.expand_dims(Ast_Seqs_eval, axis=0)),
            Ps: sess.run(tf.expand_dims(Ps_eval, axis=0)),
            Qs: sess.run(tf.expand_dims(Qs_eval, axis=0)),
        })
        print('Epoch %i: Evaluation Loss: %f ####################################################' % (
        epoch, eval_loss_val[0]))
    sess.run(data_iter_test.initializer)
    test_loss_val = sess.run([loss], feed_dict={
        # test set does not have batches, but placeholder requires batch size
        Ast_Seqs: sess.run(tf.expand_dims(Ast_Seqs_test, axis=0)),
        Ps: sess.run(tf.expand_dims(Ps_test, axis=0)),
        Qs: sess.run(tf.expand_dims(Qs_test, axis=0)),
    })
    print('Test Loss: %f ####################################################' % test_loss_val[0])