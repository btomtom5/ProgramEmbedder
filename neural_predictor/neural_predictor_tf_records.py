import os, sys, json
import tensorflow as tf
import numpy as np

from random import shuffle

from ast_tokenizer import vectorize_token_list as ast_to_sequence, NUM_TOKENS as TOKEN_DIMENSION
from matrix_learner.matrix_learner_tf_records import PRECONDITION, POSTCONDITION, AST, COND_FEATURE_LENGTH
from matrix_predictor.matrix_predictor_tf_records import MAX_SEQUENCE_LENGTH


DATA_DIR = "dev_data"


def tf_record_parser(serialized_example):
    keys_to_features = {
        "sequence": tf.FixedLenFeature([MAX_SEQUENCE_LENGTH, TOKEN_DIMENSION], tf.float32),
        "precond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.float32),
        "postcond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.float32),
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
                    'sequence': tf.train.Feature(float_list=tf.train.FloatList(value=padded_sequence.flatten())),
                    'precond': tf.train.Feature(float_list=tf.train.FloatList(value=data[PRECONDITION])),
                    'postcond': tf.train.Feature(float_list=tf.train.FloatList(value=data[POSTCONDITION])),
                }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        tf_writer.write(serialized)


def create_tf_record_from_hoare_record_paths(hoare_triples_paths, tf_record_path):
    writer = tf.python_io.TFRecordWriter(tf_record_path)
    for ht_path in hoare_triples_paths:
        with open(ht_path, 'r') as json_file:
            for line in json_file:
                write_hoare_triple_to_tf_record(line.strip(), writer)
    writer.close()
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    else:
        raise Exception("Usage Error: python3 script_name.py <data | dev_data>")

    hoare_triples = []
    for root, dirs, files in os.walk(HOARE_TRIPLES_DIR % DATA_DIR):
        for file in files:
            if file.endswith(".json"):
                print("Json file: {}".format(file))
                json_file_path = os.path.join(root, file)
                hoare_triples.append(json_file_path)
    shuffle(hoare_triples)
    train_frac, eval_frac = int(0.8 * len(hoare_triples)), int(0.9 * len(hoare_triples))
    train_triples = hoare_triples[:train_frac]
    eval_triples = hoare_triples[train_frac: eval_frac]
    test_triples = hoare_triples[eval_frac:]
    create_tf_record_from_hoare_record_paths(train_triples, TF_RECORDS_TRAIN % DATA_DIR)
    create_tf_record_from_hoare_record_paths(eval_triples, TF_RECORDS_EVAL % DATA_DIR)
    create_tf_record_from_hoare_record_paths(test_triples, TF_RECORDS_TEST % DATA_DIR)
