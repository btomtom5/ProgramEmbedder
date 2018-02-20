import tensorflow as tf


import json
from os import listdir
from os.path import isfile, join

AST_INDEX = "ast"
CONDITION_KEY = "cond"
PRECONDITION_KEY = "precond"
POSTCONDITION_KEY = "postcond"


def data_set_from_file(file_path):
    """
    Assumes file_path is of the form some/path/<Number>.json
    """
    with open(file_path) as f:
        data = json.load(f)
    precondition_tensor = tf.convert_to_tensor(data[PRECONDITION_KEY], dtype=tf.float32)
    postcondition_tensor = tf.convert_to_tensor(data[POSTCONDITION_KEY], dtype=tf.float32)
    datapoint_1 = (precondition_tensor, precondition_tensor)
    datapoint_2 = (postcondition_tensor, postcondition_tensor)
    return [datapoint_1, datapoint_2]


def data_set_from_directory(dir_path):
    """
    Assumes that the only items in dir_path are files that
    have the name: <Number>.json
    """
    dataset = []
    for item_name in listdir(dir_path):
        full_path = join(dir_path, item_name)
        if isfile(full_path):
            dataset.extend(data_set_from_file(full_path))
    return dataset


def my_input_fn(data_dirs, perform_shuffle=False, repeat_count=0, buffer_size=256, batch_size=32):
    data = []
    for dir_path in data_dirs:
        data.extend(data_set_from_directory(dir_path))
    dataset = tf.data.Dataset.from_tensors(data)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

