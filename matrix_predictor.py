# 1) train a model that reads in an AST (as DFS sequence) and then outputs a matrix
# 2) one way to approach this is by only taking the last output of the RNN......

# 3) what form is the data in? (filename=id --> Matrix, asts[id] = AST)
#   (Matrix, AST) --> (Matrix, sequence_AST)
# iterate over sequence_AST and feed it to the RNN. Save the last output of the RNN as the predicted matrix

# I want to have data in batches that I can just feed to the model

import os, sys
import tensorflow as tf
from brians_code import ast_to_sequence, MAX_SEQUENCE_LENGTH, STATEMENT_DIMENSION

from matrix_learner import H1_UNITS
from matrix_learner_tf_records import parse_ast_data, MATRICES_DIR


TF_RECORD_FILE = "Datasets/Hour of Code/matrix_predictor.tfrecord"


def cond_tf_record_parser(record):
    keys_to_features = {
        "sequence": tf.FixedLenFeature([STATEMENT_DIMENSION, MAX_SEQUENCE_LENGTH], tf.float32),
        "matrix": tf.FixedLenFeature([H1_UNITS, H1_UNITS], tf.float32),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    return parsed['precond'], parsed['postcond']


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_to_tf_record(writer, sequence, matrix):
    feature = {
        'sequence': _floats_feature(sequence),
        'matrix': _floats_feature(matrix),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


if __name__ == "main":
    ast_to_id, asts = parse_ast_data()
    writer = tf.python_io.TFRecordWriter(TF_RECORD_FILE)
    for id in range(len(asts)):
        tf.reset_default_graph()
        matrix = tf.get_variable("linear_map", shape=[H1_UNITS, H1_UNITS])
        matrix_file_path = os.path.join(MATRICES_DIR, "{}.ckpt".format(id))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, matrix_file_path)
            var_mat = matrix.eval()
            ast_seq = ast_to_sequence(asts[id])
            write_to_tf_record(writer, ast_seq, var_mat)
    writer.close()
    sys.stdout.flush()
