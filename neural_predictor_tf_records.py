import os, sys, json
import tensorflow as tf
import numpy as np

from ast_tokenizer import vectorize_token_list as ast_to_sequence, NUM_TOKENS as TOKEN_DIMENSION
from matrix_learner_tf_records import PRECONDITION, POSTCONDITION, AST, COND_FEATURE_LENGTH
from matrix_predictor import MAX_SEQUENCE_LENGTH

HOARE_TRIPLES_DIR = "Datasets/hour_of_code/dev_data/hoare_triples"
TF_RECORDS_TRAIN = "Datasets/hour_of_code/dev_data/tf_records/train.tfrecord"
TF_RECORDS_EVAL = "Datasets/hour_of_code/dev_data/tf_records/eval.tfrecord"
TF_RECORDS_TEST = "Datasets/hour_of_code/dev_data/tf_records/test.tfrecord"


def tf_record_parser(serialized_example):
    keys_to_features = {
        "sequence": tf.FixedLenFeature([MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION], tf.int64),
        "precond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64),
        "postcond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64),
    }
    parsed = tf.parse_single_example(serialized_example, keys_to_features)
    return parsed['sequence'], parsed['precond'], parsed['postcond']


def write_hoare_triple_to_tf_record(json_hoare, tf_writer):
    data = json.loads(json_hoare)
    if data.get(PRECONDITION, -1) != -1 and data.get(POSTCONDITION, -1) != -1 and data.get(AST, -1 != -1):
        sequence = ast_to_sequence(data[AST])
        amount_padding = MAX_SEQUENCE_LENGTH - sequence.shape[0]
        padded_sequence = np.pad(sequence, [[0, amount_padding], [0, 0]], mode='constant', constant_values=0)
        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list, float_list, or bytes_list
                    'sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=padded_sequence.flatten())),
                    'precond': tf.train.Feature(int64_list=tf.train.Int64List(value=data[PRECONDITION])),
                    'postcond': tf.train.Feature(int64_list=tf.train.Int64List(value=data[POSTCONDITION])),
                }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)


if __name__ == "__main__":
    writer = tf.python_io.TFRecordWriter(TF_RECORDS_FILE)
    for root, dirs, files in os.walk(HOARE_TRIPLES_DIR):
        for file in files:
            if file.endswith(".json"):
                print("Json file: {}".format(file))
                json_file_path = os.path.join(root, file)
                json_asts = []
                with open(json_file_path, 'r') as json_file:
                    for line in json_file:
                        write_hoare_triple_to_tf_record(line.strip(), writer)
    writer.close()
    sys.stdout.flush()
