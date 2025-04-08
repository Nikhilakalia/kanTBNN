import os
import random
import subprocess
import pandas as pd
import numpy as np
import tbnn.losses as losses
from functools import partial

# Base configuration settings
base_config = {
    "run_name": "duct_experiment",
    "dataset_params": {
        "file": "/home/nikhila/WDC/kan/dataset/turbulence_dataset_clean.csv",
        "Cases": [
            'squareDuctAve_Re_1100', 'squareDuctAve_Re_1150', 'squareDuctAve_Re_1250',
            'squareDuctAve_Re_1300', 'squareDuctAve_Re_1350', 'squareDuctAve_Re_1400',
            'squareDuctAve_Re_1500', 'squareDuctAve_Re_1600', 'squareDuctAve_Re_1800',
            'squareDuctAve_Re_2000', 'squareDuctAve_Re_2205', 'squareDuctAve_Re_2400',
            'squareDuctAve_Re_2600', 'squareDuctAve_Re_2900', 'squareDuctAve_Re_3200',
            'squareDuctAve_Re_3500'
        ],
        "val_set": ['squareDuctAve_Re_1300', 'squareDuctAve_Re_1800', 'squareDuctAve_Re_3200'],
        "test_set": ['squareDuctAve_Re_2000']
    },
    "training_params": {
        "loss_fn": partial(losses.aLoss, alpha=100),
        "max_epochs": 500,
        "learning_rate": 0.0005,
        "learning_rate_decay": 1.0,
        "batch_size": 64,
        "early_stopping_patience": 500,
        "early_stopping_min_delta": 1E-8
    },
    "model_params": {
        "model_type": "tbnn.models.kanTBNN",
        "width": [16, 8, 8, 8, 10],
        "grid": 8,
        "k": 3,
        "input_features": [
            'komegasst_q6', 'komegasst_q5', 'komegasst_q8', 'komegasst_I1_16',
            'komegasst_I1_7', 'komegasst_I1_3', 'komegasst_I2_6', 'komegasst_q3',
            'komegasst_I2_3', 'komegasst_I1_4', 'komegasst_I2_7', 'komegasst_I1_35',
            'komegasst_q2', 'komegasst_I1_1', 'komegasst_q4', 'komegasst_I2_8'
        ]
    }
}

# Hyperparameter ranges
width_range = range(5, 13)  # Range for width values (5 to 12 inclusive)
grid_range = range(5, 13)  # Range for grid values (5 to 12 inclusive)
learning_rate_range = [0.0001, 0.0005, 0.001, 0.002, 0.005]  # Learning rate options

# Directory for saving configurations and results
config_dir = "/home/nikhila/WDC/kan/kanTBNN/config"
base_results_dir = "/home/nikhila/WDC/kan/kanTBNN/models/multi_run"

# Ensure the config directory is in the Python path
if config_dir not in os.sys.path:
    os.sys.path.insert(0, config_dir)

# Function to calculate MSE from the results CSV file
def calculate_mse(results_file):
    try:
        df = pd.read_csv(results_file)
        mse_a11 = np.mean((df['pred_a_11'] - df['DNS_a_11']) ** 2)
        mse_a12 = np.mean((df['pred_a_12'] - df['DNS_a_12']) ** 2)
        mse_a22 = np.mean((df['pred_a_22'] - df['DNS_a_22']) ** 2)
        mse_a33 = np.mean((df['pred_a_33'] - df['DNS_a_33']) ** 2)
        mean_mse = np.mean([mse_a11, mse_a12, mse_a22, mse_a33])
        print(f"MSE for a_11: {mse_a11}")
        print(f"MSE for a_12: {mse_a12}")
        print(f"MSE for a_22: {mse_a22}")
        print(f"MSE for a_33: {mse_a33}")
        print(f"Mean MSE: {mean_mse}")
        return mean_mse
    except Exception as e:
        print(f"Error calculating MSE: {e}")
        return None

# Function to save the configuration as a Python file and run the experiment
def save_and_run_experiment(config, width1, width2, width3, grid, learning_rate, experiment_id):
    # Update hyperparameters in the config
    config["model_params"]["width"] = [16, width1, width2, width3, 10]
    config["model_params"]["grid"] = grid
    config["training_params"]["learning_rate"] = learning_rate

    # Use all 16 input features
    config["model_params"]["input_features"] = base_config["model_params"]["input_features"]
    config["run_name"] = f"duct_experiment_{experiment_id}"

    # Create directory for this experiment's results
    experiment_dir = os.path.join(base_results_dir, f"duct_experiment_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save the config as a Python file in the desired format
    config_filename = f"duct_config_{experiment_id}.py"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, "w") as f:
        f.write("import torch.nn as nn\n")
        f.write("import tbnn.losses as losses\n")
        f.write("import tbnn\n")
        f.write("from functools import partial\n")
        f.write("import os\n\n")
        f.write(f"run_name = '{config['run_name']}'\n")
        f.write(f"results_dir = '{experiment_dir}'\n")
        f.write("if not os.path.exists(results_dir): os.makedirs(results_dir)\n")
        f.write("evaluation = tbnn.evaluate.square_duct\n\n")

        # Write dataset_params
        f.write("dataset_params = " + repr(config["dataset_params"]) + "\n")

        # Write training_params
        f.write("training_params = {\n")
        f.write("    'loss_fn': partial(losses.aLoss, alpha=100),\n")
        for key, value in config["training_params"].items():
            if key != "loss_fn":
                f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")

        # Write model_params
        f.write("model_params = {\n")
        f.write("    'model_type': tbnn.models.kanTBNN,\n")
        for key, value in config["model_params"].items():
            if key != "model_type":
                f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")

        f.write("dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]\n")
        f.write("training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]\n")

    # Run the experiment using the specified shell command
    module_name = config_filename[:-3]  # Remove .py extension for module import
    cmd = f"python3 kanTBNN_training_run_config.py {module_name}"
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"Experiment {experiment_id} completed successfully.")
        print(result.stdout)

        # Locate the results CSV file and calculate MSE
        results_file = os.path.join(experiment_dir, "results.csv")  # Adjust as per actual file naming
        if os.path.exists(results_file):
            calculate_mse(results_file)
        else:
            print(f"Results file not found for experiment {experiment_id}.")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {experiment_id} failed.")
        print(e.stdout)
        print(e.stderr)

# Run experiments with random selection of hyperparameters
for i in range(1, 21):  # Run 20 random experiments
    width1 = random.choice(width_range)
    width2 = random.choice(width_range)
    width3 = random.choice(width_range)
    grid = random.choice(grid_range)
    learning_rate = random.choice(learning_rate_range)
    save_and_run_experiment(base_config.copy(), width1, width2, width3, grid, learning_rate, experiment_id=i)
