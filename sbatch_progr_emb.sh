#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3000
#SBATCH -c 32

# Add modules for OpenMind...
module add openmind/miniconda/4.0.5-python3
module add openmind/cuda/8.0
module add openmind/cudnn/8.0-5.1

# Activate conda environment
source activate prog_emb

full_clean=false
data_dir=nil
num_epochs=nil
batch_size=nil

while getopts 'f:e:b:c' opt; do
    case $opt in
        f)
            data_dir=$OPTARG
            ;;
        c)
            full_clean=true
            ;;
        e)
            num_epochs=$OPTARG
            ;;
        b)
            batch_size=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

if [ $data_dir == "nil" ]; then
    echo "usage: -f <data | dev_data> is a required parameter"
    exit 1
fi

if [ $num_epochs == "nil" ]; then
    echo "usage: -e <int> is a required parameter. Number of epochs to run the NN"
    exit 1
fi

if [ $batch_size == "nil" ]; then
    echo "usage: -b <int> is a required parameter. Size of Batch to run the NN"
    exit 1
fi

if [ "$full_clean" == "true" ]; then
    echo "performing a full clean"
    sh ./Datasets/hour_of_code/clear_data.sh -c -f ${data_dir}
    echo "re-running simulations to generate hoare triples"
    python3 Datasets/hour_of_code/simulator.py ${data_dir}
else
    echo "performing a partial clean"
    sh ./Datasets/hour_of_code/clear_data.sh -f ${data_dir}
fi

python3 matrix_learner_tf_records.py ${data_dir}

python3 matrix_learner.py ${data_dir} ${num_epochs} ${batch_size}

python3 matrix_predictor_tf_records.py ${data_dir}

python3 matrix_predictor.py ${data_dir} ${num_epochs} ${batch_size}
