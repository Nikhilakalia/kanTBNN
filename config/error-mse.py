import pandas as pd
import numpy as np

# Define the path to your CSV file
file_path_1 = '/home/nikhila/WDC/kan/kanTBNN/models/obsolete/duct_only/TBNNiii-24May27_08:45:31_df_test_tbnn_duct.csv'
file_path_2 = '/home/nikhila/WDC/kan/kanTBNN/models/duct_only/kanTBNN-25Jan30_13:12:36_df_test_tbnn_duct.csv'
file_path_3 ='/home/nikhila/WDC/kan/kanTBNN/models/multi_run/duct_experiment_3/kanTBNN-25Jan30_21:51:48_df_test_tbnn_duct.csv'
file_path_phll = "/home/nikhila/WDC/kan/kanTBNN/models/phll_only_1/kanTBNN-25Feb16_13:38:15_df_test_tbnn_phll.csv"
file_path_tbnn_phll = "/home/nikhila/WDC/kan/kanTBNN/models/obsolete/phll_only/TBNNiii-24May27_20:22:22_df_test_tbnn_phll.csv"
# Load the data
df = pd.read_csv(file_path_3)

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
