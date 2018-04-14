import os, sys, numpy as np
import tensorflow as tf
from ast_tokenizer import ast_tokenizer as ast_to_sequence, NUM_TOKENS as TOKEN_DIMENSIONS

from matrix_learner import H1_UNITS
from matrix_learner_tf_records import parse_ast_data, MATRICES_DIR


TF_RECORD_TRAIN = "Datasets/hour_of_code/data/mat_pred_train.tfrecord"
TF_RECORD_EVAL = "Datasets/hour_of_code/data/mat_pred_eval.tfrecord"
TF_RECORD_TEST = "Datasets/hour_of_code/data/mat_pred_test.tfrecord"

MAX_SEQUENCE_LENGTH = 20


def tf_record_parser(record):
    '''
    Extracts the token sequence and program matrix features from the tfrecord.
    Ads padding to the token sequence so that it is of length MAX_SEQUENCE_LENGTH.
    Flattens the matrix to 1-D (so that it is the same dimensions as the output of the
    multi-LSTM RNN model.
    Assumes the dimensions of the sequence stored in the matrix are seq_len x token_dim.
    :param record: a tfrecord
    :return: returns the sequence of dimensions MAX_SEQUENCE_LENGTH x TOKEN_DIMENSIONS
    and returns the flattened matrix which of dimensions H1_UNITS x 1
    '''
    keys_to_features = {
        "sequence": tf.FixedLenFeature([TOKEN_DIMENSIONS, MAX_SEQUENCE_LENGTH], tf.float32),
        "matrix": tf.FixedLenFeature([H1_UNITS, H1_UNITS], tf.float32),
    }

    parsed = tf.parse_single_example(record, keys_to_features)
    seq = parsed['sequence']
    amount_padding = MAX_SEQUENCE_LENGTH - tf.shape(seq)[0]
    padding = tf.constant([[0, amount_padding], [0, 0]])
    padded_seq = tf.pad(seq, padding, "CONSTANT", constant_values=0)

    mat = parsed['matrix']
    flattened_mat = tf.reshape(mat, [-1])
    return padded_seq, flattened_mat


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_to_tf_record(writer, sequence, matrix):
    feature = {
        'sequence': _floats_feature(sequence),
        'matrix': _floats_feature(matrix),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def create_tf_record_from_ast_ids(ids, file_path):
    writer = tf.python_io.TFRecordWriter(file_path)
    for id in ids:
        tf.reset_default_graph()
        matrix = tf.get_variable("linear_map", shape=[H1_UNITS, H1_UNITS])
        matrix_file_path = os.path.join(MATRICES_DIR, "{}.ckpt".format(id))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, matrix_file_path)
            var_mat = matrix.eval()
            ast_seq = ast_to_sequence(asts[id])  # assume output is STATEMENT_DIMENSION x MAX_SEQUENCE_LENGTH
            write_to_tf_record(writer, ast_seq, var_mat)
    writer.close()
    sys.stdout.flush()


if __name__ == "main":
    ast_to_id, asts = parse_ast_data()
    ids = np.random.permutation(len(ast_to_id))
    train_frac, eval_frac = int(0.8*len(ast_to_id)), int(0.9*len(ast_to_id))
    train_ids = ids[:train_frac]
    eval_ids = ids[train_frac: eval_frac]
    test_ids = ids[eval_frac:]
    create_tf_record_from_ast_ids(train_ids, TF_RECORD_TRAIN)
    create_tf_record_from_ast_ids(eval_ids, TF_RECORD_EVAL)
    create_tf_record_from_ast_ids(test_ids, TF_RECORD_TEST)

