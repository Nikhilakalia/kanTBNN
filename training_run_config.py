import pandas as pd
from tbnn.training_utils import early_stopped_tbnn_training_run, plot_loss_curve, get_dataframes, save_model
import tbnn.evaluate as evaluate
import matplotlib.pyplot as plt
import tbnn.devices as devices
device = devices.get_device()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

import sys
fullpath = '/home/ryley/WDK/ML/code/tbnn/config'
sys.path.append(fullpath)

config = __import__(args.config_file)#'CFG_sanity_check_phll_only_norealiz')
results_dir = config.results_dir
import os

dataset_params = config.dataset_params
training_params = config.training_params
model_params = config.model_params

model = model_params['model_type'](
            N=10,
            input_dim = len(model_params['input_features']),
            n_hidden = model_params['n_hidden'],
            neurons = model_params['neurons'],
            activation_function = model_params['activation_function'],
            input_feature_names=model_params['input_features']
).to(device)

sys.stdout = open(os.path.join(results_dir,f'model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.log'),'w')

df = pd.read_csv(dataset_params['file'])
df, df_train, df_valid, df_test = get_dataframes(dataset_params,print_info=True)

model, loss_vals, val_loss_vals  = early_stopped_tbnn_training_run(model = model,
                                                                   df_train = df_train,
                                                                   df_valid = df_valid,
                                                                   training_params = training_params,
                                                                   )
model.eval()
save_model(model, os.path.join(results_dir,f'model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.pickle'))

plot_loss_curve(loss_vals, 
                val_loss_vals, 
                os.path.join(results_dir,
                             f'model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.png')
                )

#evaluate.periodic_hills(model, config)
config.evaluation(model, config)
#evaluate.square_duct(model, config)

#evaluate.periodic_hills(model, config)

