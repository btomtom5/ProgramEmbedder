#!/bin/sh

# remove all the old data

rm -r data/hoare_triples
rm -r data/intermediate
rm -r data/tfrecords
rm -r data/ast_matrices
rm data/ast_to_id.txt

# create the data directories

mkdir data/hoare_triples
mkdir data/hoare_triples/hoc4
mkdir data/hoare_triples/hoc18
mkdir data/intermediate
mkdir data/tfrecords
mkdir data/ast_matrices
