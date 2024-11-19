import os
import pandas as pd
import numpy as np
import glob

# Define the directory containing the experiment results
base_results_dir = "/home/nikki/kan/kanTBNN/models/multi_run"
num_experiments = 27

# Initialize a list to store performance metrics for each experiment
results_summary = []

# Loop through each experiment
for experiment_id in range(1, num_experiments + 1):
    experiment_dir = os.path.join(base_results_dir, f"kan_experiment_{experiment_id}")
    
    # Debug: print file paths being searched
    print(f"Experiment {experiment_id} - Searching files in: {experiment_dir}")
    
    # Find the exact CSV file (assuming it's the test results file) in the experiment directory
    test_files = glob.glob(os.path.join(experiment_dir, "*_df_test_tbnn_fp.csv"))
    print("Files found:", test_files)  # Print the found files for verification
    
    if not test_files:
        print(f"Missing test file for experiment {experiment_id}. Skipping this experiment.")
        continue
    
    # Load the most recent test file based on timestamp
    test_file = max(test_files, key=os.path.getctime)
    print(f"Using file: {test_file}")  # Debug print to show which file is used
    df_test = pd.read_csv(test_file)
    
    # Columns of interest for MSE calculations
    required_dns_columns = ['DNS_a_11', 'DNS_a_12', 'DNS_a_22', 'DNS_a_33']
    required_pred_columns = ['pred_a_11', 'pred_a_12', 'pred_a_22', 'pred_a_33']
    
    # Check for required columns to avoid KeyError
    if any(col not in df_test.columns for col in required_dns_columns + required_pred_columns):
        print(f"Required DNS or predicted columns missing in test file for experiment {experiment_id}. Skipping.")
        continue

    # Calculate the MSE for each component
    mse_a11 = np.mean((df_test['pred_a_11'] - df_test['DNS_a_11'])**2)
    mse_a12 = np.mean((df_test['pred_a_12'] - df_test['DNS_a_12'])**2)
    mse_a22 = np.mean((df_test['pred_a_22'] - df_test['DNS_a_22'])**2)
    mse_a33 = np.mean((df_test['pred_a_33'] - df_test['DNS_a_33'])**2)

    # Store the results in the summary list
    results_summary.append({
        "experiment_id": experiment_id,
        "mse_a11": mse_a11,
        "mse_a12": mse_a12,
        "mse_a22": mse_a22,
        "mse_a33": mse_a33,
        "mean_mse": np.mean([mse_a11, mse_a12, mse_a22, mse_a33])  # Overall mean MSE
    })

# Convert the results to a DataFrame for easier analysis
df_results = pd.DataFrame(results_summary)

# Check if the DataFrame is empty before proceeding
if df_results.empty:
    print("No experiments found with the required data files. Please check file paths and try again.")
else:
    # Identify the best experiment based on the lowest mean MSE
    best_experiment = df_results.loc[df_results['mean_mse'].idxmin()]

    # Print out the results
    print("Best Performing Experiment:")
    print(best_experiment)
    print("\nAll Experiment Results:")
    print(df_results)

    # Save the summary DataFrame for reference
    summary_file = os.path.join(base_results_dir, "experiment_comparison_summary.csv")
    df_results.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")
