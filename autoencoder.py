import tensorflow as tf


import json
from os import listdir
from os.path import isfile, join

AST_INDEX = "ast"
PRE_CONDITION_KEY = "precond"
POST_CONDITION_KEY = "postcond"

def data_set_from_file(file_path):
    """
    Assumes file_path is of the form some/path/<Number>.json
    """

    with open(file_path) as f:
        data = json.load(f)

    data[PRE_CONDITION_KEY] = tf.convert_to_tensor(data[PRE_CONDITION_KEY], dtype=tf.float32)
    data[POST_CONDITION_KEY] = tf.convert_to_tensor(data[POST_CONDITION_KEY], dtype=tf.float32)

    return data

def get_tf_data_set(dir_path):
    """
    Assumes that the only items in dir_path are files that
    have the name: <Number>.json
    """
    data_files = [item_name for item_name in listdir(dir_path) if isfile(join(dir_path, item_name))]
    for df in data_files:


def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_directory(line):
        # 1
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:] # Last element is the label
        del parsed_line[-1] # Delete last element
        features = parsed_line # Everything (but last element) are the features
        d = dict(zip(feature_names, features)), label
        return d

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(32)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels




# 1) go through all of the files in that directory extract the precond and postcond data
# 2)