import pandas as pd
import numpy as np

# Define the path to your CSV file
file_path = '/home/nikki/kan/kanTBNN/models/obsolete/fp_only/TBNNiii-24May30_09:54:40_df_test_tbnn_fp.csv'
file_path_2 = '/home/nikki/kan/kanTBNN/models/multi_run/kan_experiment_148_opt_test/kanTBNN-24Dec02_18:31:03_df_test_tbnn_fp.csv'
file_path_3 ='/home/nikki/kan/kanTBNN/models/multi_run/kan_experiment_148_opt_test/kanTBNN-24Dec02_18:31:03_df_test_tbnn_fp.csv'

# Load the data
df = pd.read_csv(file_path_2)

# Calculate MSE for each tensor component
mse_a11 = np.mean((df['pred_a_11'] - df['DNS_a_11']) ** 2)
mse_a12 = np.mean((df['pred_a_12'] - df['DNS_a_12']) ** 2)
mse_a22 = np.mean((df['pred_a_22'] - df['DNS_a_22']) ** 2)
mse_a33 = np.mean((df['pred_a_33'] - df['DNS_a_33']) ** 2)

# Calculate mean MSE
mean_mse = np.mean([mse_a11, mse_a12, mse_a22, mse_a33])

# Output the results
print("MSE for a_11:", mse_a11)
print("MSE for a_12:", mse_a12)
print("MSE for a_22:", mse_a22)
print("MSE for a_33:", mse_a33)
print("Mean MSE:", mean_mse)
