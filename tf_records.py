import os
import tensorflow as tf
import sys
from random import shuffle
import json

AST_ID_LENGTH = 1
COND_FEATURE_LENGTH = 390
PRECONDITION = "precond"
POSTCONDITION = "postcond"
AST = "ast"

DATA_DIRECTORY = "Datasets/Hour of Code/hoare_triples"

TRAIN_TF_RECORDS_FILE = "Datasets/Hour of Code/tfrecords/train.tfrecords"
VAL_TF_RECORDS_FILE = "Datasets/Hour of Code/tfrecords/val.tfrecords"
TEST_TF_RECORDS_FILE = "Datasets/Hour of Code/tfrecords/test.tfrecords"

TRAIN_AST_FILE = "Datasets/Hour of Code/ast_data/train.ast"
VAL_AST_FILE = "Datasets/Hour of Code/ast_data/val.ast"
TEST_AST_FILE = "Datasets/Hour of Code/ast_data/test.ast"


def write_condition_to_tf_record(precond, postcond, ast_id, writer):
    feature = {
        'precond': _int64_feature(precond),
        'postcond': _int64_feature(postcond),
        'ast_id': _int64_feature([ast_id])
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_data_to_tf_record(data_files, tf_record_file, ast_info_file):
    writer = tf.python_io.TFRecordWriter(tf_record_file)
    ast_to_id = {}
    asts = []
    for json_file in data_files:
        with open(json_file) as f:
            data = json.load(f)
        if data.get(PRECONDITION, -1) != -1 and data.get(POSTCONDITION, -1) != -1 and data.get(AST, -1 != -1):
            ast_string = json.dumps(data[AST])
            if ast_to_id.get(ast_string, -1) == -1:
                asts.append(ast_string)
                ast_id = len(asts) - 1
                ast_to_id[ast_string] = ast_id
            write_condition_to_tf_record(data[PRECONDITION], data[POSTCONDITION], ast_to_id[ast_string], writer)
    writer.close()
    sys.stdout.flush()
    with open(ast_info_file, 'w+') as f:
        f.write("{}\n".format(json.dumps(ast_to_id)))
        f.write("{}\n".format(json.dumps(asts)))


def cond_tf_record_parser(record):
    keys_to_features = {
        "precond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64),
        "postcond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64),
        "ast_id": tf.FixedLenFeature([AST_ID_LENGTH], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    return parsed['precond'], parsed['postcond'], parsed['ast_id']


def parse_ast_data(file_name):
    with open(TRAIN_AST_FILE, 'r') as f:
        ast_to_id = json.loads(next(f))
        asts = json.loads(next(f))
    return ast_to_id, asts


if __name__ == "__main__":
    data_files = []
    for root, dirs, files in os.walk(DATA_DIRECTORY):
        for file in files:
            if file.endswith(".json"):
                data_file_path = os.path.join(root, file)
                data_files.append(data_file_path)
    shuffle(data_files)

    train = data_files[0:int(0.7 * len(data_files))]
    write_data_to_tf_record(train, TRAIN_TF_RECORDS_FILE, TRAIN_AST_FILE)

    val = data_files[int(0.7 * len(data_files)):int(0.85 * len(data_files))]
    write_data_to_tf_record(val, VAL_TF_RECORDS_FILE, VAL_AST_FILE)

    test = data_files[int(0.85 * len(data_files)):]
    write_data_to_tf_record(test, TEST_TF_RECORDS_FILE, TEST_AST_FILE)
