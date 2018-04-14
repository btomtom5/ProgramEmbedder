import os
import tensorflow as tf
import sys
import json

AST_ID_LENGTH = 1
COND_FEATURE_LENGTH = 390
PRECONDITION = "precond"
POSTCONDITION = "postcond"
AST = "ast"

HOARE_TRIPLES_DIR = "Datasets/hour_of_code/data/example"
INTERMEDIATE_DIR = "Datasets/hour_of_code/data/intermediate"
TF_RECORDS_DIR = "Datasets/hour_of_code/data/tfrecords"
MATRICES_DIR = "Datasets/hour_of_code/data/ast_matrices"
AST_DATA_FILE = "Datasets/hour_of_code/data/ast_data/ast_to_id.txt"


def parse_ast_data():
    with open(AST_DATA_FILE, 'r') as f:
        ast_to_id = json.loads(f.readline())
        asts = json.loads(f.readline())
    return ast_to_id, asts


def write_to_intermediate_file(inter_file, precond, postcond):
    with open(inter_file, 'a') as f:
        f.write("{}\n".format(json.dumps(precond)))
        f.write("{}\n".format(json.dumps(postcond)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def cond_tf_record_parser(record):
    keys_to_features = {
        "precond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64),
        "postcond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    return parsed['precond'], parsed['postcond']


def write_condition_to_tf_record(precond, postcond, writer):
    feature = {
        'precond': _int64_feature(precond),
        'postcond': _int64_feature(postcond),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def write_to_tf_record(tf_record_file, inter_file):
    with open(inter_file, 'r') as inter_f:
        writer = tf.python_io.TFRecordWriter(tf_record_file)
        while True:
            line1, line2 = inter_f.readline().strip(), inter_f.readline().strip()
            if not line1 or not line2:
                break
            else:
                precond = json.loads(line1)
                postcond = json.loads(line2)
                write_condition_to_tf_record(precond, postcond, writer)
        writer.close()
        sys.stdout.flush()
    print("Writing TF-Record: {}".format(tf_record_file))


if __name__ == "__main__":
    ast_to_id = {}
    asts = []
    for root, dirs, files in os.walk(HOARE_TRIPLES_DIR):
        for file in files:
            if file.endswith(".json"):
                print("Json file: {}".format(file))
                json_file_path = os.path.join(root, file)
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                if data.get(PRECONDITION, -1) != -1 and data.get(POSTCONDITION, -1) != -1 and data.get(AST, -1 != -1):
                    ast_string = json.dumps(data[AST])
                    if ast_to_id.get(ast_string, -1) == -1:
                        ast_id = len(asts)
                        asts.append(ast_string)
                        ast_to_id[ast_string] = ast_id
                    inter_name = "{}.inter".format(ast_to_id[ast_string])
                    inter_file = os.path.join(INTERMEDIATE_DIR, inter_name)
                    write_to_intermediate_file(inter_file, data[PRECONDITION], data[POSTCONDITION])

    for root, dirs, files in os.walk(INTERMEDIATE_DIR):
        for file in files:
            print("tfrecord: {}".format(file))
            ast_id, _ = os.path.splitext(file)
            ast_id = os.path.basename(ast_id)
            file_path = os.path.join(INTERMEDIATE_DIR, file)
            tf_record_name = "{}.tfrecord".format(ast_id)
            tf_record_file = os.path.join(TF_RECORDS_DIR, tf_record_name)
            write_to_tf_record(tf_record_file, file_path)

    with open(AST_DATA_FILE, 'w+') as ast_file:
        ast_file.write("{}\n".format(json.dumps(ast_to_id)))
        ast_file.write("{}\n".format(json.dumps(asts)))
