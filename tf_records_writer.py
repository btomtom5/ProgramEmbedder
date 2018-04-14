import os
import tensorflow as tf
import sys
from random import shuffle
import json

COND_FEATURE_LENGTH = 390
PRECONDITION = "precond"
POSTCONDITION = "postcond"

DATA_DIRECTORY = "Datasets/hour_of_code/hoare_triples"

TRAIN_TF_RECORDS_FILE = "Datasets/hour_of_code/tfrecords/train.tfrecords"
VAL_TF_RECORDS_FILE = "Datasets/hour_of_code/tfrecords/val.tfrecords"
TEST_TF_RECORDS_FILE = "Datasets/hour_of_code/tfrecords/test.tfrecords"


def write_condition_to_tf_record(cond, writer):
    feature = {'train/cond': _int64_feature(cond)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_data_to_tf_record(data_files, tf_record_file):
    writer = tf.python_io.TFRecordWriter(tf_record_file)
    for json_file in data_files:
        with open(json_file) as f:
            data = json.load(f)
        if data.get(PRECONDITION, -1) != -1:
            write_condition_to_tf_record(data[PRECONDITION], writer)
        if data.get(POSTCONDITION, -1) != -1:
            write_condition_to_tf_record(data[POSTCONDITION], writer)
    writer.close()
    sys.stdout.flush()


def cond_tf_record_parser(record):
    keys_to_features = {"train/cond": tf.FixedLenFeature([COND_FEATURE_LENGTH], tf.int64)}
    parsed = tf.parse_single_example(record, keys_to_features)
    return parsed["train/cond"]


if __name__ == "__main__":
    data_files = []
    for root, dirs, files in os.walk(DATA_DIRECTORY):
        for file in files:
            if file.endswith(".json"):
                data_file_path = os.path.join(root, file)
                data_files.append(data_file_path)
    shuffle(data_files)

    train = data_files[0:int(0.7 * len(data_files))]
    write_data_to_tf_record(train, TRAIN_TF_RECORDS_FILE)

    val = data_files[int(0.7 * len(data_files)):int(0.85 * len(data_files))]
    write_data_to_tf_record(val, VAL_TF_RECORDS_FILE)

    test = data_files[int(0.85 * len(data_files)):]
    write_data_to_tf_record(test, TEST_TF_RECORDS_FILE)
