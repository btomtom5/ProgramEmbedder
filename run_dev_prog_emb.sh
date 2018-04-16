#!/bin/bash

# clear and set up the environment

sh ./Datasets/hour_of_code/clear_dev_data.sh

# run python simulator

python3 Datasets/hour_of_code/simulator.py

python3 matrix_learner_tf_records.py

python3 matrix_learner.py

python3 matrix_learner_tf_records.py

python3 matrix_predictor_tf_records.py

python3 matrix_predictor.py



