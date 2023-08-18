import argparse
import ast
import configparser

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from seerocket_multivariate import fit_transform


def load_subsequence(file_path, start_index, seq_length, time_column):
    data = pd.read_csv(file_path, skiprows=range(1, start_index + 1), nrows=seq_length)
    data = data.sort_values(by=time_column, ascending=False)
    start_index += seq_length
    return data, start_index


def generate_samples(data, time_column, target_column, sequence_length):
    data[time_column] = pd.to_datetime(data[time_column])
    target_column = data.pop(target_column).to_numpy()
    time_diff = data[time_column].diff().dt.total_seconds() / 3600
    time_column = data.pop(time_column)
    split_index = time_diff[time_diff > 10].index
    data_list = np.split(data, split_index)
    target_column_list = np.split(target_column, split_index)

    samples = []
    target_value = []

    for data, target_column in zip(data_list, target_column_list):
        num_samples = len(data) - (sequence_length + 1) + 1

        for i in range(num_samples):
            sample = data[i:i + sequence_length].T
            target = target_column[i + sequence_length]
            samples.append(sample)
            target_value.append(target)

    data = np.array(samples).astype(np.float32)
    targets = np.array(target_value).astype(np.float32)

    return data, targets


def incremental_training(X_train, y_train, params, gbm_model=None):
    if gbm_model is None:
        train_data = lgb.Dataset(X_train, label=y_train)
        gbm_model = lgb.train(params, train_data)
    else:
        train_data = lgb.Dataset(X_train, label=y_train)
        gbm_model = lgb.train(params, train_data, init_model=gbm_model)
    return gbm_model


def Train(run_params, seerocket_params, LightGBM_params):
    minibatch_size = run_params["minibatch_size"]

    start_index = 0

    gbm = None

    while True:

        data, start_index = load_subsequence(run_params["train_file_path"], start_index, minibatch_size, run_params["time_column"])

        if data.empty:
            break

        data, targets = generate_samples(data, run_params["time_column"], run_params["target_column"], run_params["sequence_length"])

        features = fit_transform(data, **seerocket_params)

        gbm = incremental_training(features, targets, LightGBM_params, gbm)

    return gbm


def estimate(my_model, run_params, seerocket_params):
    data = pd.read_csv(run_params["test_file_path"])

    data = data.sort_values(by=run_params["time_column"], ascending=False)

    data, targets = generate_samples(data, run_params["time_column"], run_params["target_column"], run_params["sequence_length"])

    features = fit_transform(data, **seerocket_params)

    y_pred = my_model.predict(features)

    mae = mean_absolute_error(targets, y_pred)
    mse = mean_squared_error(targets, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(targets, y_pred)

    return mae, mse, rmse, mape


def RUN(run_params, seerocket_params, LightGBM_params):
    results = {}
    model = Train(run_params=run_params, seerocket_params=seerocket_params, LightGBM_params=LightGBM_params)
    with open(f'{run_params["save_path"]}/model.pkl', 'wb') as f:
        joblib.dump(model, f)
    mae, mse, rmse, mape = estimate(model, run_params=run_params, seerocket_params=seerocket_params)
    results[f'result'] = {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape}

    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'{run_params["save_path"]}/results.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='START')
    parser.add_argument('--config', type=str, help='path to the config file')
    args = parser.parse_args()
    config_file_path = args.config

    config = configparser.ConfigParser()
    config.read(config_file_path)


    run_params = {
        "minibatch_size": config.getint("run_params", "minibatch_size"),
        "train_file_path": config.get("run_params", "train_file_path"),
        "test_file_path": config.get("run_params", "test_file_path"),
        "save_path": config.get("run_params", "save_path"),
        "target_column": config.get("run_params", "target_column"),
        "time_column": config.get("run_params", "time_column"),
        "sequence_length": config.getint("run_params", "sequence_length")
    }

    seerocket_params = {
        "num_features": config.getint("seerocket_params", "num_features"),
        "max_dilations_per_kernel": config.getint("seerocket_params", "max_dilations_per_kernel"),
        "max_allow_num_channels": config.getint("seerocket_params", "max_allow_num_channels"),
        "kernel_size": config.getint("seerocket_params", "kernel_size"),
        "num_ones": config.getint("seerocket_params", "num_ones"),
        "dilations": ast.literal_eval(config.get("seerocket_params", "dilations")) if config.get("seerocket_params", "dilations") is not None else None,
        "initial_stride": config.getint("seerocket_params", "initial_stride"),
        "stride_increment": config.getint("seerocket_params", "stride_increment"),
        "LPPV_size": config.getint("seerocket_params", "LPPV_size"),
        "padding_left": config.getint("seerocket_params", "padding_left"),
        "padding_right": config.getint("seerocket_params", "padding_right"),
        "padding_value": config.getint("seerocket_params", "padding_value"),
        "num_kernels": int(config.get("seerocket_params", "num_kernels")) if config.get("seerocket_params", "num_kernels") != 'None' else None
    }

    LightGBM_params = {
        'objective': config.get("LightGBM_params", 'objective'),
        'boosting_type': config.get("LightGBM_params", 'boosting_type'),
        'force_col_wise': config.getboolean("LightGBM_params", 'force_col_wise'),
        'max_depth': config.getint("LightGBM_params", 'max_depth'),
        'num_leaves': config.getint("LightGBM_params", 'num_leaves'),
        'learning_rate': config.getfloat("LightGBM_params", 'learning_rate')
    }

    RUN(run_params, seerocket_params, LightGBM_params)
