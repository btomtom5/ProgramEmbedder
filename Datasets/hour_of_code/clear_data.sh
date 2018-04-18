#!/bin/sh

data_dir=dev_data
full_clean=false


while getopts ':cf:' opt; do
    case $opt in
        c)
            full_clean=true
            ;;
        f)
            data_dir=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            ;;
    esac
done


DIR_PATH=$(dirname "$0")/${data_dir}

if [ ${full_clean} == true ] ; then
    echo "cleaning the hoare triples"
    rm -r $DIR_PATH/hoare_triples
    mkdir $DIR_PATH/hoare_triples

    mkdir $DIR_PATH/hoare_triples/hoc4
    mkdir $DIR_PATH/hoare_triples/hoc18
fi

rm -r $DIR_PATH/intermediate
rm -r $DIR_PATH/tfrecords
rm -r $DIR_PATH/ast_matrices
rm $DIR_PATH/ast_to_id.txt

# create the data directories

mkdir $DIR_PATH/intermediate
mkdir $DIR_PATH/tfrecords
mkdir $DIR_PATH/ast_matrices
