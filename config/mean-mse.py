import os
import pandas as pd
import numpy as np

# Base directory where experiment folders are stored
base_dir = "/home/nikhila/WDC/kan/kanTBNN/models/multi_run"

# Variables to track best MSE and corresponding experiment
best_mse = float("inf")
best_experiment = None
best_csv_file = None

# Iterate over experiment folders
for experiment in sorted(os.listdir(base_dir)):  # Ensure sorted order
    experiment_path = os.path.join(base_dir, experiment)

    if os.path.isdir(experiment_path):
        # Find CSV file with the expected pattern
        csv_files = [f for f in os.listdir(experiment_path) if "df_test_tbnn_duct.csv" in f]

        if csv_files:
            csv_path = os.path.join(experiment_path, csv_files[0])
            df = pd.read_csv(csv_path)

            print(f"\nProcessing: {csv_path}")  # Debugging
            print("Columns in CSV:", df.columns)  # Debugging

            # Ensure required columns exist
            required_cols = ['pred_a_11', 'pred_a_12', 'pred_a_22', 'pred_a_33', 
                             'DNS_a_11', 'DNS_a_12', 'DNS_a_22', 'DNS_a_33']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"Skipping {experiment}, missing columns: {missing_cols}")
                continue

            # Compute MSE for each component
            mse_a11 = np.nanmean((df['pred_a_11'] - df['DNS_a_11']) ** 2)
            mse_a12 = np.nanmean((df['pred_a_12'] - df['DNS_a_12']) ** 2)
            mse_a22 = np.nanmean((df['pred_a_22'] - df['DNS_a_22']) ** 2)
            mse_a33 = np.nanmean((df['pred_a_33'] - df['DNS_a_33']) ** 2)

            # Compute mean MSE
            mean_mse = np.mean([mse_a11, mse_a12, mse_a22, mse_a33])

            print(f"Experiment: {experiment} - Mean MSE: {mean_mse:.10f}")  # Debugging

            # Update best MSE and corresponding experiment
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_experiment = experiment
                best_csv_file = csv_path

# Print the best experiment and its MSE
if best_experiment:
    print(f"\n--- Best Experiment Results ---")
    print(f"Best experiment: {best_experiment}")
    print(f"Best CSV file: {best_csv_file}")
    print(f"Lowest Mean MSE: {best_mse:.10f}")
else:
    print("\nNo valid CSV files found or missing required columns.")
