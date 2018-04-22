import boto3
import json
from io import BytesIO

session = boto3.Session(profile_name="brian_personal_aws")
resource = session.resource('s3')

data_store_name = 'program-embedder'
data_store = resource.Bucket(data_store_name)

# TODO create a timeit decorator to measure the bottlenecks of the system.


def load_original_data(prefix="data/dev_data/"):
    # iterate through the original hoc4 and hoc18 datasets
    for file_summary in data_store.objects.filter(Prefix=prefix):
        if len(file_summary.key) > 0 and file_summary.key[-1] == "/":
            continue
        else:
            try:
                decoded_json = file_summary.get()['Body'].read().decode('utf-8')
                yield json.loads(decoded_json)
            except UnicodeDecodeError:
                print("Warning skipped file: {}".format(file_summary.key))


def write_hoare_triples(basepath, hoare_triples, batch_number):
    dumped_data = BytesIO(json.dumps(hoare_triples).encode())
    data_store.upload_fileobj(dumped_data, basepath + str(batch_number) + ".json")


# HOC DATA
HOC4_DEV_PATH = "data/dev_data/hoc4_batched"
HOC18_DEV_PATH = "data/dev_data/hoc18_batched"
HOC4_FULL_PATH = "data/full_data/hoc4_batched"
HOC18_FULL_PATH = "data/full_data/hoc18_batched"

# HOARE TRIPLES
HOARE_TRIPLES_DEV_DIR = "data/dev_data/hoare_triples/"
HOARE_TRIPLES_FULL_DIR = "data/full_data/hoare_triples/"

# MATRIX LEARNER TF RECORDS
INTERMEDIATE_DIR = "Datasets/hour_of_code/%s/intermediate"
TF_RECORDS_DIR = "Datasets/hour_of_code/%s/tfrecords"
MATRICES_DIR = "Datasets/hour_of_code/%s/ast_matrices"
AST_DATA_FILE = "Datasets/hour_of_code/%s/ast_to_id.txt"

# Matrix Predictor TFRecord
TF_RECORD_TRAIN = "Datasets/hour_of_code/%s/tfrecords/mat_pred_train.tfrecord"
TF_RECORD_EVAL = "Datasets/hour_of_code/%s/tfrecords/mat_pred_eval.tfrecord"
TF_RECORD_TEST = "Datasets/hour_of_code/%s/tfrecords/mat_pred_test.tfrecord"

# Neural Predictor Tf Records
TF_RECORDS_TRAIN = "Datasets/hour_of_code/%s/hoare_triples_tf_records/train.tfrecord"
TF_RECORDS_EVAL = "Datasets/hour_of_code/%s/hoare_triples_tf_records/eval.tfrecord"
TF_RECORDS_TEST = "Datasets/hour_of_code/%s/hoare_triples_tf_records/test.tfrecord"

# Neural predictor logging outputs
NEURAL_PREDICTOR_TRAIN_LOGS = "job_results/%s/neural_predictor_train_log.txt"
NEURAL_PREDICTOR_EVAL_LOGS = "job_results/%s/neural_predictor_eval_log.txt"
NEURAL_PREDICTOR_TEST_LOGS = "job_results/%s/neural_predictor_test_log.txt"

# Matrix predictor logging output
MATRIX_PREDICTOR_TRAIN_LOGS = "job_results/%s/matrix_predictor_train_log.txt"
MATRIX_PREDICTOR_EVAL_LOGS = "job_results/%s/matrix_predictor_eval_log.txt"
MATRIX_PREDICTOR_TEST_LOGS = "job_results/%s/matrix_predictor_test_log.txt"



