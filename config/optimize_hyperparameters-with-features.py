import os
import optuna
import pandas as pd
import numpy as np
import random
import subprocess
import glob
import matplotlib.pyplot as plt
from functools import partial

# Define paths
base_results_dir = "/home/nikki/kan/kanTBNN/models/multi_run"
config_dir = "/home/nikki/kan/kanTBNN/config"
error_calcs_script = os.path.join(config_dir, "error_calcs.py")
venv_activate = "/home/nikki/kan/kan_project/bin/activate"  # Path to virtual environment activation script

# Full list of available input features
available_features = [
    'komegasst_I1_1', 'komegasst_I1_3', 'komegasst_I1_4', 'komegasst_I1_5', 
    'komegasst_I1_7', 'komegasst_I1_9', 'komegasst_I1_10', 'komegasst_I1_12', 
    'komegasst_I1_13', 'komegasst_I1_16', 'komegasst_q5', 'komegasst_q6'
]

# Load previous results if available
initial_trials = []
for i in range(1, 43):  # Load up to experiment 42 for initial trials
    if i < 28:
        result_file = os.path.join(base_results_dir, f"kan_experiment_{i}", f"kanTBNN-{i}_df_test_tbnn_fp.csv")
    else:
        result_file = os.path.join(base_results_dir, f"kan_experiment_{i}_opt", f"kanTBNN-{i}_df_test_tbnn_fp.csv")
    
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        mean_mse = np.mean([
            np.mean((df['pred_a_11'] - df['DNS_a_11'])**2),
            np.mean((df['pred_a_12'] - df['DNS_a_12'])**2),
            np.mean((df['pred_a_22'] - df['DNS_a_22'])**2),
            np.mean((df['pred_a_33'] - df['DNS_a_33'])**2)
        ])
        initial_trials.append({
            'middle_width': 5 + (i % 3) * 2,
            'grid': 3 + (i % 3),
            'learning_rate': 0.001 * (1 + i % 3),
            'objective': mean_mse
        })

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
    config_filename = f"NIKKI_config_fp_hyper_{experiment_id}_opt.py"
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
def run_experiment_and_get_mse(config, experiment_id, feature_tracker):
    experiment_dir = os.path.join(base_results_dir, f"kan_experiment_{experiment_id}_opt")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the experiment configuration
    config_path = save_experiment_config(config, experiment_id)
    module_name = os.path.basename(config_path)[:-3]

    # Run the experiment using the virtual environment and configuration file
    cmd = f"source {venv_activate} && python3 /home/nikki/kan/kanTBNN/kanTBNN_training_run_config.py {module_name}"
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

    # Update feature tracker
    for feature in config["model_params"]["input_features"]:
        feature_tracker[feature] += 1

    mse_results = calculate_mse(experiment_dir)
    return mse_results['mean_mse']

# Objective function for Optuna
def objective(trial, feature_tracker):
    experiment_id = 43 + trial.number  # Start from 43 onward for new trials
    middle_width = trial.suggest_int("middle_width", 5, 10)
    grid = trial.suggest_int("grid", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    input_features = random.sample(available_features, 6)

    config = {
        "run_name": f"kan_experiment_{experiment_id}_opt",
        "results_dir": os.path.join(base_results_dir, f"kan_experiment_{experiment_id}_opt"),
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
            "width": [6, middle_width, 10],
            "grid": grid,
            "k": 3,
            "input_features": input_features,
        }
    }

    mean_mse = run_experiment_and_get_mse(config, experiment_id, feature_tracker)
    return mean_mse

# Initialize the study and populate it with initial trials
study = optuna.create_study(direction="minimize")
feature_tracker = {feature: 0 for feature in available_features}  # Initialize feature tracker

for trial in initial_trials:
    study.enqueue_trial(trial)

study.optimize(lambda trial: objective(trial, feature_tracker), n_trials=58)  # 58 new trials to reach 100 total

# Plot feature usage and save it
plt.figure(figsize=(10, 6))
plt.bar(feature_tracker.keys(), feature_tracker.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Usage Count")
plt.title("Input Feature Usage Across Trials")
plt.tight_layout()
plt.savefig(os.path.join(base_results_dir, "input_feature_usage.png"))

# Output best results
print("Best hyperparameters:", study.best_params)
print("Best mean MSE:", study.best_value)
