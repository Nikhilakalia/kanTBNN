import pandas as pd
from tbnn.training_utils import early_stopped_training_run, plot_loss_curve, get_dataframes, save_model
import tbnn.evaluate as evaluate
import matplotlib.pyplot as plt
import tbnn.devices as devices
from evaluation_config import evaluate_model_with_config
device = devices.get_device()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

import sys
fullpath = '/scratch/niki/kan/kanTBNN/config'
sys.path.append(fullpath)

config = __import__(args.config_file)
results_dir = config.results_dir
import os

dataset_params = config.dataset_params
training_params = config.training_params
model_params = config.model_params

model = model_params['model_type'](
            width = model_params['width'],
            grid = model_params['grid'],
            k = model_params['k'],
            seed = 7,
            input_feature_names=model_params['input_features']
).to(device)

sys.stdout = open(os.path.join(results_dir,f'{model.barcode}.log'),'w')

df, df_train, df_valid, df_test = get_dataframes(dataset_params,print_info=True)

model, loss_vals, val_loss_vals  = early_stopped_training_run(model = model,
                                                                   df_train = df_train,
                                                                   df_valid = df_valid,
                                                                   training_params = training_params,
                                                                   results_dir = results_dir
                                                                   )
model.eval()
save_model(model, os.path.join(results_dir,f'{model.barcode}.pickle'))

plot_loss_curve(loss_vals, 
                val_loss_vals, 
                os.path.join(results_dir,
                             f'{model.barcode}_losses.png')
                )

evaluate_model_with_config(model.barcode,config)
#evaluate.periodic_hills(model, config)
#if config.evaluation is not None:
#    config.evaluation(model, config)
#evaluate.square_duct(model, config)

#evaluate.periodic_hills(model, config)


