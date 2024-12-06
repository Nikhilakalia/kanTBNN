import os
import optuna
import pandas as pd
import numpy as np
import subprocess
import glob
import csv
from functools import partial

# Define paths for Cedar
base_results_dir = "/home/niki/kan/models/multi_run_square_duct"
config_dir = "/home/niki/kan/kanTBNN/config"
venv_activate = "/home/niki/kan/kan_project/bin/activate"  # Path to virtual environment activation script

# Input features for the square duct case
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

# Check the total number of features
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

# Function to save configuration as a Python file
def save_experiment_config(config, experiment_id):
    config_filename = f"NIKKI_config_square_duct_{experiment_id}.py"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, "w") as f:
        f.write("import torch.nn as nn\n")
        f.write("import tbnn.losses as losses\n")
        f.write("import tbnn\n")
        f.write("from functools import partial\n")
        f.write("import os\n\n")
        f.write(f"run_name = '{config['run_name']}'\n")
        f.write(f"results_dir = '{config['results_dir']}'\n")
        f.write("evaluation = tbnn.evaluate.square_duct\n\n")

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
    experiment_dir = os.path.join(base_results_dir, f"kan_experiment_square_duct_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the experiment configuration
    config_path = save_experiment_config(config, experiment_id)
    module_name = os.path.basename(config_path)[:-3]

    # Run the experiment using the virtual environment
    cmd = f"source {venv_activate} && python3 /home/niki/kan/kanTBNN/kanTBNN_training_run_config.py {module_name}"
    subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')

    mse_results = calculate_mse(experiment_dir)
    
    # Save results for analysis
    with open(os.path.join(base_results_dir, "results_square_duct.csv"), mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([experiment_id, config["model_params"]["width"], config["model_params"]["grid"], config["training_params"]["learning_rate"], mse_results["mean_mse"]])

    return mse_results['mean_mse']

# Objective function for Optuna
def objective(trial):
    experiment_id = trial.number + 1  # Use sequential numbering for IDs
    width_1 = trial.suggest_int("width_1", 8, 15)
    width_2 = trial.suggest_int("width_2", 8, 15)
    width_3 = trial.suggest_int("width_3", 8, 15)
    width_4 = trial.suggest_int("width_4", 8, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.001)

    config = {
        "run_name": f"kan_experiment_square_duct_{experiment_id}",
        "results_dir": os.path.join(base_results_dir, f"kan_experiment_square_duct_{experiment_id}"),
        "dataset_params": {
            "file": "/home/niki/kan/data/turbulence_dataset_clean.csv",
            "Cases": [
                'squareDuctAve_Re_1100', 'squareDuctAve_Re_1150', 
                'squareDuctAve_Re_1300', 'squareDuctAve_Re_1400', 
                'squareDuctAve_Re_1600', 'squareDuctAve_Re_1800', 
                'squareDuctAve_Re_2000', 'squareDuctAve_Re_2400', 
                'squareDuctAve_Re_3200', 'squareDuctAve_Re_3500'
            ],
            "val_set": ['squareDuctAve_Re_1300', 'squareDuctAve_Re_1800', 'squareDuctAve_Re_3200'],
            "test_set": ['squareDuctAve_Re_2000']
        },
        "training_params": {
            "max_epochs": 500,
            "learning_rate": learning_rate,
            "learning_rate_decay": 1.0,
            "batch_size": 32,
            "early_stopping_patience": 500,
            "early_stopping_min_delta": 1E-8,
        },
        "model_params": {
            "model_type": "tbnn.models.kanTBNN",
            "width": [16, width_1, width_2, width_3, width_4],
            "grid": 9,
            "k": 3,
            "input_features": all_features,
        }
    }

    mean_mse = run_experiment_and_get_mse(config, experiment_id)
    return mean_mse

# Initialize the study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Adjust number of trials as needed

# Output best results
print("Best hyperparameters:", study.best_params)
print("Best mean MSE:", study.best_value)
