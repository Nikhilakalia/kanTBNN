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

def evaluate_model_with_config(config):
    model_TBNN = torch.load(os.path.join(config.results_dir,f'{config.barcode_TBNN}.pickle')).to(device)
    model_KCNN = torch.load(os.path.join(config.results_dir,f'{config.barcode_KCNN}.pickle')).to(device)

    sys.stdout = open(os.path.join(config.results_dir,f'{config.barcode_TBNN}_{config.barcode_KCNN}_injection_evaluation.log'),'w')
    model_TBNN.eval()
    model_KCNN.eval()

    if config.evaluation is not None:
        if isinstance(config.evaluation,list):
            for eval in config.evaluation:
                    eval(model_TBNN, model_KCNN, config)
        else:
            config.evaluation(model_TBNN, model_KCNN, config)

if __name__ == "__main__":
    fullpath = '/home/ryley/WDK/ML/code/tbnn/config'
    sys.path.append(fullpath)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()
    config = __import__(args.config_file)
    evaluate_model_with_config(config)


