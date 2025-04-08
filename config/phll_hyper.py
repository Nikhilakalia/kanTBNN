import os
import pandas as pd
import numpy as np
import random
import subprocess
import csv

# Define paths
base_results_dir = "/home/nikki/kan/kanTBNN/models/multi_run_phll"
config_dir = "/home/nikki/kan/kanTBNN/config"
venv_activate = "/home/nikki/kan/kan_project/bin/activate"
results_csv = os.path.join(base_results_dir, "phll_random_results.csv")

# Ensure results directory exists
os.makedirs(base_results_dir, exist_ok=True)

# Full list of input features
input_features = [
    'komegasst_q6', 'komegasst_q5', 'komegasst_q8', 'komegasst_I1_16',
    'komegasst_I1_7', 'komegasst_I1_3', 'komegasst_I2_6', 'komegasst_q3',
    'komegasst_I2_3', 'komegasst_I1_4', 'komegasst_I2_7', 'komegasst_I1_35',
    'komegasst_q2', 'komegasst_q7', 'komegasst_I1_1', 'komegasst_q4',
    'komegasst_I2_8'
]

# Initialize results CSV if not exists
if not os.path.exists(results_csv):
    with open(results_csv, "w") as f:
        f.write("experiment_id,width_1,width_2,width_3,grid,learning_rate,mse\n")

# Function to calculate MSE
def calculate_mse(experiment_dir):
    csv_files = [f for f in os.listdir(experiment_dir) if f.endswith("_df_test_tbnn_duct.csv")]
    if not csv_files:
        return None
    df = pd.read_csv(os.path.join(experiment_dir, csv_files[0]))
    try:
        mse_a11 = np.mean((df['pred_a_11'] - df['DNS_a_11']) ** 2)
        mse_a12 = np.mean((df['pred_a_12'] - df['DNS_a_12']) ** 2)
        mse_a22 = np.mean((df['pred_a_22'] - df['DNS_a_22']) ** 2)
        mse_a33 = np.mean((df['pred_a_33'] - df['DNS_a_33']) ** 2)
        return np.mean([mse_a11, mse_a12, mse_a22, mse_a33])
    except KeyError:
        print(f"Missing columns in {csv_files[0]}, skipping MSE calculation.")
        return None

# Function to save the configuration file
def save_experiment_config(config, experiment_id):
    config_filename = f"phll_config_{experiment_id}.py"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, "w") as f:
        f.write("import torch.nn as nn\n")
        f.write("import tbnn.losses as losses\n")
        f.write("import tbnn\n")
        f.write("from functools import partial\n")
        f.write("import os\n\n")
        f.write(f"run_name = '{config['run_name']}'\n")
        f.write(f"results_dir = '{config['results_dir']}'\n")
        f.write("evaluation = tbnn.evaluate.periodic_hills\n\n")
        f.write("dataset_params = " + repr(config["dataset_params"]) + "\n\n")
        f.write("training_params = {\n")
        f.write("    'loss_fn': partial(losses.aLoss, alpha=100),\n")
        for key, value in config["training_params"].items():
            f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")
        f.write("model_params = {\n")
        f.write("    'model_type': tbnn.models.kanTBNN,\n")
        for key, value in config["model_params"].items():
            if key != "model_type":
                f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")
        f.write("dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]\n")
        f.write("training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]\n")
    return config_path

# Function to run experiment
def run_experiment_and_get_mse(config, experiment_id):
    experiment_dir = os.path.join(base_results_dir, f"phll_experiment_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)

    config_path = save_experiment_config(config, experiment_id)
    module_name = os.path.basename(config_path)[:-3]

    try:
        cmd = f"source {venv_activate} && python3 /home/nikki/kan/kanTBNN/kanTBNN_training_run_config.py {module_name}"
        subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        return calculate_mse(experiment_dir)
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_id} failed: {e}")
        return None

# Run 10 random experiments
for experiment_id in range(1, 11):  # Running 10 experiments
    width_1 = random.randint(8, 15)
    width_2 = random.randint(8, 15)
    width_3 = random.randint(8, 15)
    grid = random.randint(5, 10)
    learning_rate = random.uniform(0.000005, 0.00001)  # Adjusted learning rate

    config = {
        "run_name": f"phll_experiment_{experiment_id}",
        "results_dir": os.path.join(base_results_dir, f"phll_experiment_{experiment_id}"),
        "dataset_params": {
            "file": "/home/nikki/kan/data/turbulence_dataset_clean.csv",
            "Cases": ['case_0p5', 'case_0p8', 'case_1p0', 'case_1p2', 'case_1p5'],
            "val_set": ['case_0p8'],
            "test_set": ['case_1p2'],
        },
        "training_params": {
            "max_epochs": 20000,
            "learning_rate": learning_rate,
            "learning_rate_decay": 1.0,
            "batch_size": 128,
            "early_stopping_patience": 5000,
            "early_stopping_min_delta": 1E-8,
        },
        "model_params": {
            "model_type": "tbnn.models.kanTBNN",
            "width": [17, width_1, width_2, width_3, 10],
            "grid": grid,
            "k": 3,
            "input_features": input_features,
        }
    }

    mse = run_experiment_and_get_mse(config, experiment_id)
    if mse is not None:
        with open(results_csv, "a") as f:
            f.write(f"{experiment_id},{width_1},{width_2},{width_3},{grid},{learning_rate},{mse}\n")
        print(f"Experiment {experiment_id}: MSE = {mse}")
    else:
        with open(results_csv, "a") as f:
            f.write(f"{experiment_id},{width_1},{width_2},{width_3},{grid},{learning_rate},NaN\n")
        print(f"Experiment {experiment_id} failed.")
