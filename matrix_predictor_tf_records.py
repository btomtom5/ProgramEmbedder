import os, sys, numpy as np, json
import tensorflow as tf
from ast_tokenizer import vectorize_token_list as ast_to_sequence, NUM_TOKENS as TOKEN_DIMENSIONS

from matrix_learner import H1_UNITS
from matrix_learner_tf_records import parse_ast_data, MATRICES_DIR, AST_DATA_FILE

# <<<<<<<<<<<<<<<<<<<<<<<<<< TODO: CHANGE FILES BACK TO FULL DATA SET >>>>>>>>>>>>>>
# TF_RECORD_TRAIN = "Datasets/hour_of_code/data/tfrecords/mat_pred_train.tfrecord"
# TF_RECORD_EVAL = "Datasets/hour_of_code/data/tfrecords/mat_pred_eval.tfrecord"
# TF_RECORD_TEST = "Datasets/hour_of_code/data/tfrecords/mat_pred_test.tfrecord"

TF_RECORD_TRAIN = "Datasets/hour_of_code/%s/tfrecords/mat_pred_train.tfrecord"
TF_RECORD_EVAL = "Datasets/hour_of_code/%s/tfrecords/mat_pred_eval.tfrecord"
TF_RECORD_TEST = "Datasets/hour_of_code/%s/tfrecords/mat_pred_test.tfrecord"

MAX_SEQUENCE_LENGTH = 20
DATA_DIR = None


def tf_record_parser(serialized_example):
    '''
    Extracts the token sequence and program matrix features from the tfrecord.
    Flattens the matrix to 1-D (so that it is the same dimensions as the output of the
    multi-LSTM RNN model.
    Assumes the dimensions of the sequence stored in the matrix are seq_len x token_dim.
    :param serialized_example: a tfrecord
    :return: returns the sequence of dimensions MAX_SEQUENCE_LENGTH x TOKEN_DIMENSIONS
    and returns the flattened matrix which of dimensions H1_UNITS x 1
    '''
    keys_to_features = {
        "sequence": tf.FixedLenFeature([MAX_SEQUENCE_LENGTH, TOKEN_DIMENSIONS], tf.int64),
        "matrix": tf.FixedLenFeature([H1_UNITS**2], tf.float32),
        "ast_id": tf.FixedLenFeature([1], tf.int64)
    }
    parsed = tf.parse_single_example(serialized_example, keys_to_features)
    seq = parsed['sequence']
    mat = parsed['matrix']
    ast_id = parsed['ast_id']
    return seq, mat, ast_id


def write_to_tf_record(writer, sequence, matrix, id):
    '''
    Creates an Example Feature in the tfrecord that @writer writes to.
    Adds padding to sequence so that it is of dimensions MAX_SEQUENCE_LENGTH x TOKEN_DIMENSIONS
    :param writer: a tfrecord writer object that writers to a predefined file
    :param sequence: the serialized AST expressed as 1-hot token encodings
    :param matrix: the matrix 'leared' by running matrix_predictor.
    :param id: the canonical id for each AST in the hoare triples
    :return: NOTHING
    '''
    # construct the Example proto boject
    amount_padding = MAX_SEQUENCE_LENGTH - sequence.shape[0]
    padded_sequence = np.pad(sequence, [[0, amount_padding], [0, 0]], mode='constant', constant_values=0)
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features contains a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list, float_list, or bytes_list
                'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=padded_sequence.flatten())),
                'matrix': tf.train.Feature(float_list=tf.train.FloatList(value=matrix.flatten())),
                'ast_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[id]))
            }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)


def create_tf_record_from_ast_ids(matrices_file_path, ids, write_file_path):
    writer = tf.python_io.TFRecordWriter(write_file_path)
    for id in ids:
        tf.reset_default_graph()
        matrix = tf.get_variable("linear_map", shape=[H1_UNITS, H1_UNITS])
        matrix_file_path = os.path.join(matrices_file_path, "{}.ckpt".format(id))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, matrix_file_path)
            var_mat = matrix.eval()
            ast_seq = ast_to_sequence(json.loads(asts[id]))  # assume output is STATEMENT_DIMENSION x MAX_SEQUENCE_LENGTH
            write_to_tf_record(writer, ast_seq, var_mat, id)
    writer.close()
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        raise Exception("Usage Error: python3 script_name.py <data | dev_data>")

    ast_to_id, asts = parse_ast_data(AST_DATA_FILE % DATA_DIR)
    ids = np.random.permutation(len(ast_to_id))
    train_frac, eval_frac = int(0.8*len(ast_to_id)), int(0.9*len(ast_to_id))
    train_ids = ids[:train_frac]
    eval_ids = ids[train_frac: eval_frac]
    test_ids = ids[eval_frac:]
    create_tf_record_from_ast_ids(MATRICES_DIR % DATA_DIR, train_ids, TF_RECORD_TRAIN % DATA_DIR)
    create_tf_record_from_ast_ids(MATRICES_DIR % DATA_DIR, eval_ids, TF_RECORD_EVAL % DATA_DIR)
    create_tf_record_from_ast_ids(MATRICES_DIR % DATA_DIR, test_ids, TF_RECORD_TEST % DATA_DIR)

