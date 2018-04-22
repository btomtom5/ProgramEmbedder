import glob
import os
from os.path import isfile, join

import json


def generate_filepaths(file_dir):
    return glob.glob(file_dir + "/*.json")


hoc4_path = "./data/hoc4/asts"
hoc4_output = "./data/hoc4_batched"
hoc4_files = generate_filepaths(hoc4_path)

hoc18_path = "./data/hoc18/asts"
hoc18_output = "./data/hoc18_batched"
hoc18_files = generate_filepaths(hoc18_path)

if not os.path.isdir(hoc18_output):
    os.mkdir(hoc18_output)

if not os.path.isdir(hoc4_output):
    os.mkdir(hoc4_output)

print("hoc 18 files: %d" % len(hoc18_files))
print("hoc 4 files: %d" % len(hoc4_files))

print("example file name: %s" % hoc4_files[0])
print("example file name: %s" % hoc18_files[0])

print("example array to be dumped:")
print("\t" + str(json.loads("[{\"hello\": \"world\"}, {\"bye\": \"world\"}]")))


def batch_files(files, output_dir, batch_size=1000):
    batch_number = 0
    count = 0
    batched_data = []
    for file_name in files:
        with open(file_name, 'r') as file:
            batched_data.append(json.load(file))
            count += 1

        if count == batch_size:
            write_to_file(batch_number, batched_data, output_dir)

            count = 0
            batch_number += 1
            batched_data = []

    write_to_file(batch_number, batched_data, output_dir)


def write_to_file(batch_number, batched_data, output_dir):
    with open(join(output_dir, str(batch_number) + ".json"), "w+") as new_batch_file:
        new_batch_file.write(json.dumps(batched_data))


batch_files(hoc4_files, hoc4_output)
batch_files(hoc18_files, hoc18_output)
