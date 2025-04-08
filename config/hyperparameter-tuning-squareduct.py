import os
import optuna
import pandas as pd
import numpy as np
import subprocess
from functools import partial

# Define paths
base_results_dir = "/home/nikhila/WDC/kan/kanTBNN/models/multi_run"
config_dir = "/home/nikhila/WDC/kan/kanTBNN/config"
venv_activate = "/home/nikhila/WDC/kan/kan_project/bin/activate"
bo_results_csv = "/home/nikhila/WDC/kan/kanTBNN/bo_results.csv"

# Full list of input features
input_features = [
    'komegasst_I1_1', 'komegasst_I1_3', 'komegasst_I1_4', 'komegasst_I1_5',
    'komegasst_I1_7', 'komegasst_I1_9', 'komegasst_I1_10', 'komegasst_I1_12',
    'komegasst_I1_13', 'komegasst_I1_16', 'komegasst_q5', 'komegasst_q6',
    'komegasst_q2', 'komegasst_q3', 'komegasst_q4', 'komegasst_I2_6'
]

# Initialize BO results CSV
if not os.path.exists(bo_results_csv):
    with open(bo_results_csv, "w") as f:
        f.write("experiment_id,middle_width_1,middle_width_2,middle_width_3,grid,learning_rate,mse\n")

# Helper function to calculate MSE from experiment-specific CSV file
def calculate_mse_from_experiment_dir(experiment_dir):
    try:
        for file_name in os.listdir(experiment_dir):
            if file_name.endswith("_df_test_tbnn_duct.csv"):
                csv_path = os.path.join(experiment_dir, file_name)
                df = pd.read_csv(csv_path)
                mse_a11 = np.mean((df['pred_a_11'] - df['DNS_a_11']) ** 2)
                mse_a12 = np.mean((df['pred_a_12'] - df['DNS_a_12']) ** 2)
                mse_a22 = np.mean((df['pred_a_22'] - df['DNS_a_22']) ** 2)
                mse_a33 = np.mean((df['pred_a_33'] - df['DNS_a_33']) ** 2)
                return np.mean([mse_a11, mse_a12, mse_a22, mse_a33])
        print(f"No matching CSV file found in {experiment_dir}")
    except Exception as e:
        print(f"Error calculating MSE in {experiment_dir}: {e}")
    return float('inf')  # Return high loss if MSE cannot be calculated

# Extract hyperparameters from config file
def extract_hyperparameters_from_config(config_file):
    hyperparams = {}
    try:
        with open(config_file, "r") as f:
            for line in f:
                if "'width':" in line:
                    exec(f"width={line.split(':', 1)[1].strip().strip(',')} ", {}, hyperparams)
                elif "'grid':" in line:
                    exec(f"grid={line.split(':', 1)[1].strip().strip(',')} ", {}, hyperparams)
                elif "'learning_rate':" in line:
                    exec(f"learning_rate={line.split(':', 1)[1].strip().strip(',')} ", {}, hyperparams)
        return {
            "middle_width_1": hyperparams["width"][1],
            "middle_width_2": hyperparams["width"][2],
            "middle_width_3": hyperparams["width"][3],
            "grid": hyperparams["grid"],
            "learning_rate": hyperparams["learning_rate"]
        }
    except Exception as e:
        print(f"Error extracting hyperparameters from {config_file}: {e}")
        return None

# Load initial trials for warm start
def load_initial_trials():
    trials = []
    for folder_name in os.listdir(base_results_dir):
        experiment_dir = os.path.join(base_results_dir, folder_name)
        if os.path.isdir(experiment_dir):
            mse = calculate_mse_from_experiment_dir(experiment_dir)
            config_file = os.path.join(config_dir, f"{folder_name}.py")
            if mse != float('inf') and os.path.exists(config_file):
                hyperparams = extract_hyperparameters_from_config(config_file)
                if hyperparams:
                    trials.append(optuna.trial.create_trial(
                        params={
                            "middle_width_1": hyperparams["middle_width_1"],
                            "middle_width_2": hyperparams["middle_width_2"],
                            "middle_width_3": hyperparams["middle_width_3"],
                            "grid": hyperparams["grid"],
                            "learning_rate": hyperparams["learning_rate"],
                        },
                        distributions={
                            "middle_width_1": optuna.distributions.IntDistribution(5, 10),
                            "middle_width_2": optuna.distributions.IntDistribution(5, 10),
                            "middle_width_3": optuna.distributions.IntDistribution(5, 10),
                            "grid": optuna.distributions.IntDistribution(5, 10),
                            "learning_rate": optuna.distributions.FloatDistribution(0.0003, 0.0005),
                        },
                        value=mse
                    ))
                    print(f"Loaded trial from {experiment_dir}: MSE = {mse}")
    return trials

# Save configuration as Python file
def save_experiment_config(config, experiment_id):
    config_filename = f"NIKKI_config_duct_hyper_{experiment_id}_opt.py"
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

# Run experiment and get MSE
def run_experiment_and_get_mse(config, experiment_id):
    experiment_dir = os.path.join(base_results_dir, f"kan_experiment_{experiment_id}_opt")
    os.makedirs(experiment_dir, exist_ok=True)
    config_path = save_experiment_config(config, experiment_id)
    module_name = os.path.basename(config_path)[:-3]
    try:
        cmd = f"source {venv_activate} && python3 /home/nikhila/WDC/kan/kanTBNN/kanTBNN_training_run_config.py {module_name}"
        subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        return calculate_mse_from_experiment_dir(experiment_dir)
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_id} failed with error: {e}")
        return float('inf')

# Objective function for Optuna
def objective(trial):
    experiment_id = 20 + trial.number
    middle_width_1 = trial.suggest_int("middle_width_1", 15, 20)
    middle_width_2 = trial.suggest_int("middle_width_2", 15, 20)
    middle_width_3 = trial.suggest_int("middle_width_3", 15, 20)
    grid = trial.suggest_int("grid", 10, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.0003, 0.0005)

    config = {
        "run_name": f"kan_experiment_{experiment_id}_opt",
        "results_dir": os.path.join(base_results_dir, f"kan_experiment_{experiment_id}_opt"),
        "dataset_params": {
            "file": "/home/nikhila/WDC/kan/dataset/turbulence_dataset_clean.csv",
            "Cases": [
                'squareDuctAve_Re_1100', 'squareDuctAve_Re_1150', 'squareDuctAve_Re_1250',
                'squareDuctAve_Re_1300', 'squareDuctAve_Re_1350', 'squareDuctAve_Re_1400',
                'squareDuctAve_Re_1500', 'squareDuctAve_Re_1600', 'squareDuctAve_Re_1800',
                'squareDuctAve_Re_2000', 'squareDuctAve_Re_2205', 'squareDuctAve_Re_2400',
                'squareDuctAve_Re_2600', 'squareDuctAve_Re_2900', 'squareDuctAve_Re_3200', 'squareDuctAve_Re_3500'
            ],
            "val_set": ['squareDuctAve_Re_1300', 'squareDuctAve_Re_1800', 'squareDuctAve_Re_3200'],
            "test_set": ['squareDuctAve_Re_2000']
        },
        "training_params": {
            "max_epochs": 500,
            "learning_rate": learning_rate,
            "batch_size": 64,
            "early_stopping_patience": 500,
            "early_stopping_min_delta": 1E-8,
            "learning_rate_decay": 1.0,
        },
        "model_params": {
            "model_type": "tbnn.models.kanTBNN",
            "width": [16, middle_width_1, middle_width_2, middle_width_3, 10],
            "grid": grid,
            "k": 3,
            "input_features": input_features,
        }
    }

    mse = run_experiment_and_get_mse(config, experiment_id)
    if mse != float('inf'):
        print(f"Experiment {experiment_id}: MSE = {mse}")
        with open(bo_results_csv, "a") as f:
            f.write(f"{experiment_id},{middle_width_1},{middle_width_2},{middle_width_3},{grid},{learning_rate},{mse}\n")
    else:
        print(f"Experiment {experiment_id} failed, continuing optimization.")
    return mse

# Load warm start trials and run Bayesian Optimization
study = optuna.create_study(direction="minimize")
warm_start_trials = load_initial_trials()
for trial in warm_start_trials:
    study.add_trial(trial)

study.optimize(objective, n_trials=20)

print("\n=== Optimization Complete ===")
print(f"Best MSE: {study.best_value}")
print(f"Best Params: {study.best_params}")
