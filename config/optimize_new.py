import os
import optuna
import pandas as pd
import numpy as np
import subprocess
import glob
import matplotlib.pyplot as plt
from functools import partial
import csv

# Define paths
base_results_dir = "/home/nikki/kan/kanTBNN/models/multi_run_v2"
config_dir = "/home/nikki/kan/kanTBNN/config"
venv_activate = "/home/nikki/kan/kan_project/bin/activate"  # Path to virtual environment activation script

# All 16 features
# Define all available features
all_features = [
    'komegasst_q6',
    'komegasst_q5',
    'komegasst_q8',
    'komegasst_I1_16',
    'komegasst_I1_7',
    'komegasst_I1_3',
    'komegasst_I2_6',
    'komegasst_q3',
    'komegasst_I2_3',
    'komegasst_I1_4',
    'komegasst_I2_7',
    'komegasst_I1_35',
    'komegasst_q2',
    'komegasst_I1_1',
    'komegasst_q4',
    'komegasst_I2_8',
]

# Count the total number of features
print(f"All features: {all_features}")
print(f"Total number of features: {len(all_features)}")

# Helper function to calculate MSE based on experiment results
def calculate_mse(experiment_dir):
    csv_files = glob.glob(os.path.join(experiment_dir, "*_df_test_tbnn_fp.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {experiment_dir} matching '*_df_test_tbnn_fp.csv'")
    df_pred = pd.read_csv(csv_files[0])
    mse_a11 = np.mean((df_pred['pred_a_11'] - df_pred['DNS_a_11'])**2)
    mse_a12 = np.mean((df_pred['pred_a_12'] - df_pred['DNS_a_12'])**2)
    mse_a22 = np.mean((df_pred['pred_a_22'] - df_pred['DNS_a_22'])**2)
    mse_a33 = np.mean((df_pred['pred_a_33'] - df_pred['DNS_a_33'])**2)
    mean_mse = np.mean([mse_a11, mse_a12, mse_a22, mse_a33])
    return {"mse_a11": mse_a11, "mse_a12": mse_a12, "mse_a22": mse_a22, "mse_a33": mse_a33, "mean_mse": mean_mse}

# Function to save configuration as a Python file and run the experiment
def save_experiment_config(config, experiment_id):
    config_filename = f"new_config_fp_hyper_{experiment_id}_opt.py"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, "w") as f:
        f.write("import torch.nn as nn\n")
        f.write("import tbnn.losses as losses\n")
        f.write("import tbnn\n")
        f.write("from functools import partial\n")
        f.write("import os\n\n")
        f.write(f"run_name = '{config['run_name']}'\n")
        f.write(f"results_dir = '{config['results_dir']}'\n")
        f.write("evaluation = tbnn.evaluate.flatplate\n\n")

        f.write("dataset_params = " + repr(config["dataset_params"]) + "\n")
        f.write("training_params = {\n")
        f.write("    'loss_fn': partial(losses.aLoss),\n")
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

# Function to run the experiment and calculate MSE
def run_experiment_and_get_mse(config, experiment_id):
    experiment_dir = os.path.join(base_results_dir, f"kan_experiment_v2_{experiment_id}_opt")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the experiment configuration
    config_path = save_experiment_config(config, experiment_id)
    module_name = os.path.basename(config_path)[:-3]

    # Run the experiment using the virtual environment and configuration file
    cmd = f"source {venv_activate} && python3 /home/nikki/kan/kanTBNN/kanTBNN_training_run_config.py {module_name}"
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

    mse_results = calculate_mse(experiment_dir)
    
    # Save results for analysis
    with open(os.path.join(base_results_dir, "results_v2.csv"), mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([experiment_id, config["model_params"]["width"], config["model_params"]["grid"], config["training_params"]["learning_rate"], mse_results["mean_mse"]])

    return mse_results['mean_mse']

# Objective function for Optuna
def objective(trial):
    experiment_id = trial.number + 1  # Use sequential numbering for IDs
    width_1 = trial.suggest_int("width_1", 8, 10)
    width_2 = trial.suggest_int("width_2", 10, 12)
    learning_rate = trial.suggest_float("learning_rate", 0.002, 0.005)

    config = {
        "run_name": f"kan_experiment_v2_{experiment_id}_opt",
        "results_dir": os.path.join(base_results_dir, f"kan_experiment_v2_{experiment_id}_opt"),
        "dataset_params": {
            "file": "/home/nikki/kan/data/turbulence_dataset_clean.csv",
            "Cases": ['fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'],
            "val_set": ['fp_3030', 'fp_1410', 'fp_4060'],
            "test_set": ['fp_3630']
        },
        "training_params": {
            "max_epochs": 8000,
            "learning_rate": learning_rate,
            "learning_rate_decay": 1.0,
            "batch_size": 64,
            "early_stopping_patience": 500,
            "early_stopping_min_delta": 1E-8,
        },
        "model_params": {
            "model_type": "tbnn.models.kanTBNN",
            "width": [16, width_1, width_2, 10],
            "grid": 4,
            "k": 4,
            "input_features": all_features,
        }
    }

    mean_mse = run_experiment_and_get_mse(config, experiment_id)
    return mean_mse

# Initialize the study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)  # Number of trials

# Output best results
print("Best hyperparameters:", study.best_params)
print("Best mean MSE:", study.best_value)
