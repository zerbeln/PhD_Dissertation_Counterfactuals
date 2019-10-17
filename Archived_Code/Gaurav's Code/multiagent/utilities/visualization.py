import time

# private data starting with _
_current_run_timestamp = time.time()
_logs_prefix = "logs"
_train_folder = "train"
_eval_folder = "eval"
_type_string = "global"


def train_path():
    return _logs_prefix + "/11/" + str(int(_current_run_timestamp)) + "/" + _train_folder + "/" + _type_string


def eval_path():
    return _logs_prefix + "/1/" + str(int(_current_run_timestamp)) + "/" + _eval_folder + "/" + _type_string
