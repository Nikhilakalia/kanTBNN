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

config = __import__(args.config_file)
results_dir = config.results_dir
import os

model_params = config.model_params

model = model_params['model_type'](model_dict=model_params)
sys.stdout = open(os.path.join(results_dir,f'{model.barcode}.log'),'w')
config.evaluation(model, config)


