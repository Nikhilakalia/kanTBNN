import pandas as pd
from tbnn.training_utils import early_stopped_training_run, plot_loss_curve, get_dataframes, save_model
import tbnn.evaluate as evaluate
import matplotlib.pyplot as plt
import tbnn.devices as devices
device = devices.get_device()
import argparse

import torch

import sys

import os

def evaluate_model_with_config(barcode,config):
    results_dir = config.results_dir
    model = torch.load(os.path.join(results_dir,f'{barcode}.pickle')).to(device)
    sys.stdout = open(os.path.join(results_dir,f'{model.barcode}_evaluation.log'),'w')
    model.eval()
    if config.evaluation is not None:
        if isinstance(config.evaluation,list):
            for eval in config.evaluation:
                    eval(model, config)
        else:
            config.evaluation(model, config)

if __name__ == "__main__":
    fullpath = '/home/ryley/WDK/ML/code/tbnn/config'
    sys.path.append(fullpath)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("barcode")
    args = parser.parse_args()
    config = __import__(args.config_file)
    evaluate_model_with_config(args.barcode,config)


