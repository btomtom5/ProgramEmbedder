#!/bin/sh

# remove all the old data

DIR_PATH=$(dirname "$0")

echo $DIR_PATH

rm -r $DIR_PATH/dev_data/hoare_triples
rm -r $DIR_PATH/dev_data/intermediate
rm -r $DIR_PATH/dev_data/tfrecords
rm -r $DIR_PATH/dev_data/ast_matrices
rm $DIR_PATH/dev_data/ast_to_id.txt

# create the data directories

mkdir $DIR_PATH/dev_data/hoare_triples
mkdir $DIR_PATH/dev_data/hoare_triples/hoc4
mkdir $DIR_PATH/dev_data/hoare_triples/hoc18
mkdir $DIR_PATH/dev_data/intermediate
mkdir $DIR_PATH/dev_data/tfrecords
mkdir $DIR_PATH/dev_data/ast_matrices


