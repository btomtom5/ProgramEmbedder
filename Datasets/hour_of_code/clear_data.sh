#!/bin/sh

# remove all the old data

DIR_PATH=$(dirname "$0")

echo $DIR_PATH

rm -r $DIR_PATH/data/hoare_triples
rm -r $DIR_PATH/data/intermediate
rm -r $DIR_PATH/data/tfrecords
rm -r $DIR_PATH/data/ast_matrices
rm $DIR_PATH/data/ast_to_id.txt

# create the data directories

mkdir $DIR_PATH/data/hoare_triples
mkdir $DIR_PATH/data/hoare_triples/hoc4
mkdir $DIR_PATH/data/hoare_triples/hoc18
mkdir $DIR_PATH/data/intermediate
mkdir $DIR_PATH/data/tfrecords
mkdir $DIR_PATH/data/ast_matrices
