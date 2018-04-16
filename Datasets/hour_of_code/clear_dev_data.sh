#!/bin/sh

# remove all the old data

rm -r dev_data/hoare_triples
rm -r dev_data/intermediate
rm -r dev_data/tfrecords
rm -r dev_data/ast_matrices
rm dev_data/ast_to_id.txt

# create the data directories

mkdir dev_data/hoare_triples
mkdir dev_data/hoare_triples/hoc4
mkdir dev_data/hoare_triples/hoc18
mkdir dev_data/intermediate
mkdir dev_data/tfrecords
mkdir dev_data/ast_matrices


