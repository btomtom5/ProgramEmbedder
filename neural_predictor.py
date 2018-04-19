import tensorflow as tf

from matrix_learner_tf_records import COND_FEATURE_LENGTH
from matrix_predictor_tf_records import MAX_SEQUENCE_LENGTH
from ast_tokenizer import NUM_TOKENS as TOKEN_DIMENSION
from neural_predictor_tf_records import tf_record_parser, TF_RECORDS_TRAIN, TF_RECORDS_EVAL, TF_RECORDS_TEST


NUM_EPOCHS = 5
BATCH_SIZE = 8
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
Ast_Seqs_eval, Ps_eval, Qs_eval = data_iter_eval.get_next()

data_iter_test = tf.data.TFRecordDataset(TF_RECORDS_TEST)\
            .map(tf_record_parser)\
            .make_initializable_iterator()
Ast_Seqs_test, Ps_test, Qs_test = data_iter_test.get_next()


def multi_lstm_model():
    cells = [tf.nn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True) for size in LSTM_CELLS]
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)


def post_lstm_fc_layer(lstm_output):
    num_weights_in_pred_NN = 0
    for i in range(1, len(PREDICTED_NN_LAYERS)):
        num_layer_weights = PREDICTED_NN_LAYERS[i-1]*PREDICTED_NN_LAYERS[i]
        num_layer_biases = PREDICTED_NN_LAYERS[i]
        num_weights_in_pred_NN += num_layer_weights + num_layer_biases
    weights = tf.Variable(tf.random_normal([LSTM_CELLS[-1], num_weights_in_pred_NN]), name='post_lstm_weights')
    biases = tf.Variable(tf.random_normal([num_weights_in_pred_NN]), name='post_lstm_biases')
    output = tf.nn.relu(tf.add(tf.matmul(lstm_output, weights), biases))
    return output


def reshape_concatenated_weights_to_layers(concatenated_weights, preconds):
    '''
    Takes in a hidden state (output from an LSTM model).
    Reshapes the hidden_state into a neural network with layers described
    by PREDICTED_NN_LAYERS. The first layer is the input size and the last layer
    is the output size. The middle layers are hidden layers.
    Hidden layers use ReLU.
    Assumes that PREDICTED_NN_LAYERS has at least 2 layer sizes

    :param hidden_state: the (concatenated) weights of the "predicted" feed-forward
    network that will represent functionality of the AST on the precond --> postcond
    :return: the outputs of the feed-forward network (sigmoid)
    '''
    start = 0
    end = 0
    layer = preconds
    indeces = list(range(1, len(PREDICTED_NN_LAYERS)))
    non_linear_transforms = [tf.nn.sigmoid]*(len(PREDICTED_NN_LAYERS) - 1) #[tf.nn.relu]*(len(PREDICTED_NN_LAYERS) - 2) + [tf.nn.sigmoid]
    for i, transform in zip(indeces, non_linear_transforms):
        end += PREDICTED_NN_LAYERS[i-1]*PREDICTED_NN_LAYERS[i]
        weights = tf.reshape(
            tf.gather(concatenated_weights, list(range(start, end)), axis=1),
            [-1, PREDICTED_NN_LAYERS[i-1], PREDICTED_NN_LAYERS[i]]
        )
        start = end
        end += PREDICTED_NN_LAYERS[i]
        biases = tf.reshape(
            tf.gather(concatenated_weights, list(range(start, end)), axis=1),
            [-1, PREDICTED_NN_LAYERS[i]]
        )
        # tf.einsum refers to Einstein Summation notation. Here 'bi,bij->bj' refers
        # to batch matrix multiplication.
        layer = transform(tf.add(tf.einsum('bi,bij->bj', layer, weights), biases))
        start = end
    return layer


def binary_threshold(x):
    cond = tf.less(x, tf.ones(tf.shape(x))*0.5)
    thresholded_x = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return thresholded_x


# Define the model (in sequential order pretty much)
Ast_Seqs = tf.placeholder(tf.float32, [None, MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION], name="Ast_Seqs")
lstm_model = multi_lstm_model()
all_outputs, _ = tf.nn.dynamic_rnn(lstm_model, Ast_Seqs, dtype=tf.float32)
final_output = tf.unstack(tf.gather(all_outputs, [MAX_SEQUENCE_LENGTH - 1], axis=1), axis=1)[0]
nn_as_vec = post_lstm_fc_layer(final_output)
Ps = tf.placeholder(tf.float32, [None, COND_FEATURE_LENGTH], name="Ps")
Qs_pred = reshape_concatenated_weights_to_layers(nn_as_vec, Ps)
Qs = tf.placeholder(tf.float32, [None, COND_FEATURE_LENGTH], name="Qs")

loss = tf.losses.mean_squared_error(Qs, Qs_pred)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

labels = tf.placeholder(tf.float32, [COND_FEATURE_LENGTH], name="labels")
predictions = tf.placeholder(tf.float32, [COND_FEATURE_LENGTH], name="predictions")
accuracy, accuracy_op = tf.metrics.accuracy(
    labels=binary_threshold(labels),
    predictions=binary_threshold(predictions),
    name="Q_pred_accuracy"
)
accuracy_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="Q_pred_accuracy")

init = tf.global_variables_initializer()
local_vars_init = tf.variables_initializer(var_list=accuracy_vars)
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
        sess.run(data_iter_eval.initializer)
        eval_loss_val, eval_preds = sess.run([loss, Qs_pred], feed_dict={
            # evaluation set does not have batches, but placeholder requires batch size
            Ast_Seqs: sess.run(tf.expand_dims(Ast_Seqs_eval, axis=0)),
            Ps: sess.run(tf.expand_dims(Ps_eval, axis=0)),
            Qs: sess.run(tf.expand_dims(Qs_eval, axis=0)),
        })
        sess.run([local_vars_init, accuracy_op], feed_dict={
            labels: sess.run(Qs_eval),
            predictions: sess.run(tf.unstack(eval_preds, axis=0)[0])
        })
        print('Epoch %i:\tEvaluation Loss: %f\tTest Accuracy: %f ####################################################' % (
        epoch, eval_loss_val, sess.run(accuracy)))
    sess.run(data_iter_test.initializer)
    test_loss_val, test_preds = sess.run([loss, Qs_pred], feed_dict={
        # test set does not have batches, but placeholder requires batch size
        Ast_Seqs: sess.run(tf.expand_dims(Ast_Seqs_test, axis=0)),
        Ps: sess.run(tf.expand_dims(Ps_test, axis=0)),
        Qs: sess.run(tf.expand_dims(Qs_test, axis=0)),
    })
    sess.run([local_vars_init, accuracy_op], feed_dict={
        labels: sess.run(Qs_test),
        predictions: sess.run(tf.unstack(test_preds, axis=0)[0])
    })
    print('Test Loss: %f\tTest Accuracy: %f ####################################################' % (
        test_loss_val, sess.run(accuracy)))