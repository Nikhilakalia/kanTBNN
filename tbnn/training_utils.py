import pandas as pd
import torch
import numpy as np
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import ParameterSampler
import copy
import tbnn.losses as losses
import tbnn.dataloaders as dataloaders
import tbnn.models as models
import tbnn.evaluate as evaluate

import tbnn.devices as devices
device = devices.get_device()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=1E-8):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def count_nonrealizable(b):
    n_nr = torch.count_nonzero(losses.realizabilityPenalty(b))
    return n_nr

def early_stopped_tbnn_training_run(model, df_train, df_valid, training_params): #data_loader = dataloaders.aDataset):
    loss_fn = training_params['loss_fn']
    data_loader, mseLoss = get_dataloader_type(loss_fn)
    loss_values = []
    val_loss_values = []

    tDs = data_loader(df_train, input_features=model.input_feature_names)
    vDs = data_loader(df_valid, input_features=model.input_feature_names, scaler_X = tDs.scaler_X)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_params['learning_rate_decay'])
    early_stopper = EarlyStopper(patience=training_params['early_stopping_patience'], min_delta=training_params['early_stopping_min_delta'])

    print_table_header()

    for epoch in range(1, training_params['max_epochs']+1):
        model.train()
        for inputs, labels in loader:
            y_pred, g_pred = model(*inputs) #(X_batch, #Tn_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, *labels) #Sometimes, labels is just b, sometimes, it is k and a (depends on dataLoader)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            loss, val_loss = evaluate.intermediate_model(model,[tDs,vDs], loss_fn)

        loss_values.append(loss)
        val_loss_values.append(val_loss)

        if val_loss_values[-1] < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)

        if (epoch % 10==0 or epoch==1) or (early_stopper.early_stop(val_loss_values[-1])):
            evaluate.print_intermediate_info(model,[tDs,vDs],loss_fn,mseLoss,epoch,lr_scheduler._last_lr[-1])
            
        if early_stopper.early_stop(val_loss_values[-1]):
            break   
        
        lr_scheduler.step()
    print_table_footer()
    return best_model, loss_values, val_loss_values

def plot_loss_curve(loss_vals, val_loss_vals, filename):
    fig, ax = plt.subplots(1,figsize=(5,5))
    ax.plot(loss_vals,'-',color='b',label='Training')
    ax.plot(val_loss_vals,'--',color='r',label='Validation')
    ax.semilogy()
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(filename,dpi=300)

def get_dataframes(dataset_params,print_info = False):
    df = pd.read_csv(dataset_params['file'])
    df = df[df['Case'].isin(dataset_params['Cases'])]
    df_train = df[~df['Case'].isin(dataset_params['test_set']+dataset_params['val_set'])].copy()
    df_valid = df[df['Case'].isin(dataset_params['val_set'])].copy()
    df_test = df[df['Case'].isin(dataset_params['test_set'])].copy()

    memorization_flag = False
    if len(df_train)==0:
        print(f'=====> WARNING: Training dataset length: {len(df_train)}, assuming memorization test!')
        df_train=df_valid
        memorization_flag = True
    if print_info:
        print(f'========== Dataset info ==========')
        print(f'Dataset params: {dataset_params}')
        print(f'Train/Valid/Test points: {len(df_train)} / {len(df_valid)} / {len(df_test)}')
        print(f'Memorization?: {memorization_flag}')
    return df, df_train, df_valid, df_test

def print_table_header():
    print('EPOCH     LR        TRAIN         VALID             MSE:T/V                  RL:T/V             %NR_t/%NR_v')
    print('===============================================================================================================')
    return

def print_table_footer():
    print('===============================================================================================================')
    return

def get_dataloader_type(loss_fn):
    if loss_fn.func == losses.aLoss:
        data_loader = dataloaders.aDataset
        mseLoss = losses.mseLoss_khat
    elif loss_fn.func == losses.bLoss:
        data_loader = dataloaders.bDataset
        mseLoss = losses.mseLoss
    else:
        raise LookupError(f'Unknown loss function: {loss_fn}')
    return data_loader, mseLoss

def save_model(model, filename):
    torch.save(model, filename)
    print(f'Saved model to {filename}')
    return 


