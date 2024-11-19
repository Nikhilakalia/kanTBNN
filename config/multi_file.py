import os
import random
import subprocess
from functools import partial
import tbnn.losses as losses
import tbnn
import itertools

# Base configuration settings without `loss_fn` in `training_params`
base_config = {
    "run_name": "kan_experiment",
    "dataset_params": {
        "file": "/home/nikki/kan/data/turbulence_dataset_clean.csv",
        "Cases": ['fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'],
        "val_set": ['fp_3030', 'fp_1410', 'fp_4060'],
        "test_set": ['fp_3630']
    },
    "training_params": {
        # `loss_fn` will be added directly in the generated .py file
        "max_epochs": 6000,
        "learning_rate": 0.002,
        "learning_rate_decay": 1.0,
        "batch_size": 64,
        "early_stopping_patience": 100000,
        "early_stopping_min_delta": 1E-8,
    },
    "model_params": {
        "model_type": "tbnn.models.kanTBNN",  # Placeholder to be replaced in .py file
        "width": [6, 5, 10],  # Placeholder; middle layer width will be varied
        "grid": 5,
        "k": 3,
        "input_features": []  # Placeholder; features will be selected dynamically
    }
}

# Full list of available input features
available_features = [
    'komegasst_I1_1', 'komegasst_I1_3', 'komegasst_I1_4', 'komegasst_I1_5', 
    'komegasst_I1_7', 'komegasst_I1_9', 'komegasst_I1_10', 'komegasst_I1_12', 
    'komegasst_I1_13', 'komegasst_I1_16', 'komegasst_q5', 'komegasst_q6'
]

# Hyperparameter options
middle_width_options = [5, 7, 8]  # Middle layer width options
grid_options = [3, 4, 5]
learning_rates = [0.001, 0.002, 0.005]

# Directory for saving configurations and results
config_dir = "/home/nikki/kan/kanTBNN/config"
base_results_dir = "/home/nikki/kan/kanTBNN/models/multi_run"  # Updated to "multi_run"

# Ensure the config directory is in the Python path
if config_dir not in os.sys.path:
    os.sys.path.insert(0, config_dir)

# Function to save the configuration as a Python file and run the experiment
def save_and_run_experiment(config, middle_width, grid, learning_rate, experiment_id):
    # Update hyperparameters in the config
    config["model_params"]["width"] = [6, middle_width, 10]
    config["model_params"]["grid"] = grid
    config["training_params"]["learning_rate"] = learning_rate

    # Randomly select 6 input features
    config["model_params"]["input_features"] = random.sample(available_features, 6)
    config["run_name"] = f"kan_experiment_{experiment_id}"

    # Create directory for this experiment's results
    experiment_dir = os.path.join(base_results_dir, f"kan_experiment_{experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)  # Ensure results_dir exists

    # Save the config as a Python file in the desired format
    config_filename = f"NIKKI_config_fp_hyper_{experiment_id}.py"
    config_path = os.path.join(config_dir, config_filename)
    with open(config_path, "w") as f:
        f.write("import torch.nn as nn\n")
        f.write("import tbnn.losses as losses\n")
        f.write("import tbnn\n")
        f.write("from functools import partial\n")
        f.write("import os\n\n")
        f.write(f"run_name = '{config['run_name']}'\n")
        f.write(f"results_dir = '{experiment_dir}'\n")  # Save results in the specific experiment directory
        f.write("if not os.path.exists(results_dir): os.makedirs(results_dir)\n")  # Ensure results_dir exists
        f.write("evaluation = tbnn.evaluate.flatplate\n\n")

        # Write dataset_params
        f.write("dataset_params = " + repr(config["dataset_params"]) + "\n")
        
        # Write training_params with `loss_fn` as partial(losses.aLoss)
        f.write("training_params = {\n")
        f.write("    'loss_fn': partial(losses.aLoss),\n")
        for key, value in config["training_params"].items():
            f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")

        # Write model_params, with `model_type` directly assigned to the class `tbnn.models.kanTBNN`
        f.write("model_params = {\n")
        f.write("    'model_type': tbnn.models.kanTBNN,\n")  # Reference the actual class
        for key, value in config["model_params"].items():
            if key != "model_type":  # `model_type` is already written
                f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")

        f.write("dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]\n")
        f.write("training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]\n")

    # Run the experiment using the specified shell command
    module_name = config_filename[:-3]  # Remove .py extension for module import
    cmd = f"python3 kanTBNN_training_run_config.py {module_name}"
    subprocess.run(cmd, shell=True, check=True)

# Run experiments for all combinations of hyperparameters
for i, (middle_width, grid, lr) in enumerate(itertools.product(middle_width_options, grid_options, learning_rates), start=1):
    save_and_run_experiment(base_config.copy(), middle_width, grid, lr, experiment_id=i)
