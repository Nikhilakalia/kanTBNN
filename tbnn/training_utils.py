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

import tbnn.devices as devices
device = devices.get_device()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def count_nonrealizable(b):
    n_nr = torch.count_nonzero(losses.realizabilityPenalty(b))
    return n_nr

def early_stopped_tbnnPlus_training_run(model, training_params, df_tv, data_loader = dataloaders.bDataset, loss_fn = losses.bLoss):
    loss_values = []
    val_loss_values = []
    df_valid = df_tv[df_tv['Case'].isin(training_params['val_set'])]
    df_train = df_tv[~df_tv['Case'].isin(training_params['val_set'])]
    tDs = data_loader(df_train, input_features=model.input_feature_names)
    vDs = data_loader(df_valid, input_features=model.input_feature_names, scaler_X = tDs.scaler_X)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])
    print(f'Training points: {len(df_train)}, validation points {len(df_valid)}')
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_params['learning_rate_decay'])
    early_stopper = EarlyStopper(patience=training_params['early_stopping_patience'], min_delta=1E-6)

    print('EPOCH    LR        TRAIN     VALID         MSE_b:T/V         MSE_g1:T/V            RL:T/V         %NR_t/%NR_v')
    print('=============================================================================================================')
    for epoch in range(1, training_params['max_epochs']+1):
        model.train()
        for X_batch, T_batch, y_batch in loader:
            y_pred, g_pred = model(X_batch, T_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, T, y in DataLoader(tDs, shuffle=False, batch_size=tDs.__len__()):
                y_pred_train, g_pred = model(X, T)
                loss_values.append(loss_fn(y_pred_train,y).item())
                mse_b_t = losses.mseLoss(y_pred_train,y).item()  
                rl_t = losses.realizabilityLoss(y_pred_train).item()  
                #mse_g1tilde_t = losses.g1tildeLoss(g_pred, g1tilde)

            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__()):
                y_pred_val, g_pred = model(X, T)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_b_v = losses.mseLoss(y_pred_val,y).item()  
                rl_v = losses.realizabilityLoss(y_pred_val).item() 
                #mse_g1tilde_v = losses.g1tildeLoss(g_pred, g1tilde)

        if val_loss_values[-1] < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)
        if (epoch % 10==0 or epoch==1) or (early_stopper.early_stop(val_loss_values[-1])):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_b_t:.5f} / {mse_b_v:.5f}   "
                  #f"{mse_g1tilde_t:.5f} / {mse_g1tilde_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")
            
        if early_stopper.early_stop(val_loss_values[-1]):
            break   

        lr_scheduler.step()
    return best_model, loss_values, val_loss_values

def early_stopped_tbnn_training_run(model, training_params, df_tv, data_loader = dataloaders.bDataset, loss_fn = losses.bLoss):
    loss_values = []
    val_loss_values = []
    df_valid = df_tv[df_tv['Case'].isin(training_params['val_set'])]
    df_train = df_tv[~df_tv['Case'].isin(training_params['val_set'])]
    tDs = data_loader(df_train, input_features=model.input_feature_names)
    vDs = data_loader(df_valid, input_features=model.input_feature_names, scaler_X = tDs.scaler_X)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])
    print(f'Training points: {len(df_train)}, validation points {len(df_valid)}')
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_params['learning_rate_decay'])
    early_stopper = EarlyStopper(patience=training_params['early_stopping_patience'], min_delta=1E-6)

    print('EPOCH    LR        TRAIN     VALID         MSE:T/V              RL:T/V         %NR_t/%NR_v')
    print('=============================================================================================')
    for epoch in range(1, training_params['max_epochs']+1):
        model.train()
        for X_batch, T_batch, y_batch in loader:
            y_pred, g_pred = model(X_batch, T_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, T, y in DataLoader(tDs, shuffle=False, batch_size=tDs.__len__()):
                y_pred_train, g_pred = model(X, T)
                loss_values.append(loss_fn(y_pred_train,y).item())
                mse_b_t = losses.mseLoss(y_pred_train,y).item()  
                rl_t = losses.realizabilityLoss(y_pred_train).item()  

            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__()):
                y_pred_val, g_pred = model(X, T)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_b_v = losses.mseLoss(y_pred_val,y).item()  
                rl_v = losses.realizabilityLoss(y_pred_val).item() 

        if val_loss_values[-1] < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)
        if (epoch % 10==0 or epoch==1) or (early_stopper.early_stop(val_loss_values[-1])):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_b_t:.5f} / {mse_b_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")
            
        if early_stopper.early_stop(val_loss_values[-1]):
            break   
        
        lr_scheduler.step()
    return best_model, loss_values, val_loss_values

"""
def early_stopped_training_run(model, training_params, df_tv, data_loader = dataloaders.bDataset, loss = losses.bLoss):
    if data_loader.isinstance(dataloaders.bDatasetPlus):
        training_tbnnPlus = True
    loss_fn = loss
    loss_values = []
    val_loss_values = []
    df_valid = df_tv[df_tv['Case'].isin(training_params['val_set'])]
    df_train = df_tv[~df_tv['Case'].isin(training_params['val_set'])]

    tDs = data_loader(df_train, input_features=model.input_feature_names)
    if training_tbnnPlus:
        vDs = data_loader(df_valid, input_features=model.input_feature_names, scaler_X = tDs.X_scaler, scaler_g1tilde = tDs.g1tilde_scaler)
    else:
        vDs = data_loader(df_valid, input_features=model.input_feature_names, scaler_X = tDs.X_scaler)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])
    print(f'Training points: {len(df_train)}, validation points {len(df_valid)}')
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_params['learning_rate_decay'])
    early_stopper = EarlyStopper(patience=training_params['early_stopping_patience'], min_delta=1E-6)

    print('EPOCH    LR        TRAIN     VALID         MSE:T/V              RL:T/V         %NR_t/%NR_v')
    print('=============================================================================================')
    for epoch in range(1, training_params['max_epochs']+1):
        model.train()
        if training_tbnnPlus:
            for X_batch, T_batch, y_batch, g1tilde_batch in loader:
                y_pred, g_pred = model(X_batch, T_batch)
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_batch, g1tilde_batch)
                loss.backward()
                optimizer.step()
        else:
            for X_batch, T_batch, y_batch in loader:
                y_pred, g_pred = model(X_batch, T_batch)
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, T, y in DataLoader(tDs, shuffle=False, batch_size=tDs.__len__())[0:3]:
                y_pred_train, g_pred = model(X, T)
                loss_values.append(loss_fn(y_pred_train,y).item())
                mse_t = losses.mseLoss(y_pred_train,y).item()  
                rl_t = losses.realizabilityLoss(y_pred_train).item()  

            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__())[0:3]:
                y_pred_val, g_pred = model(X, T)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_v = losses.mseLoss(y_pred_val,y).item()  
                rl_v = losses.realizabilityLoss(y_pred_val).item() 

        if val_loss_values[-1] < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)
        if early_stopper.early_stop(val_loss_values[-1]):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_t:.5f} / {mse_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")
            break   

        if (epoch % 10==0 or epoch==1):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_t:.5f} / {mse_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")

        lr_scheduler.step()
    return best_model, loss_values, val_loss_values

def early_stopped_tbnn_training_run(model_params, training_params, df_tv):
    loss_fn = losses.bLoss
    loss_values = []
    val_loss_values = []
    df_valid = df_tv[df_tv['Case'].isin(training_params['val_set'])]
    df_train = df_tv[~df_tv['Case'].isin(training_params['val_set'])]
    tDs = dataloaders.bDataset(df_train, input_features=model_params['input_features'])
    vDs = dataloaders.bDataset(df_valid, input_features=model_params['input_features'], scaler = tDs.X_scaler)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])

    print(f'Training points: {len(df_train)}, validation points {len(df_valid)}')

    model = models.TBNN(N = 10,
                 input_dim = len(model_params['input_features']),
                 n_hidden = model_params['n_hidden'],
                 neurons = model_params['neurons'],
                 activation_function = model_params['activation_function']
                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_params['learning_rate_decay'])
    early_stopper = EarlyStopper(patience=training_params['early_stopping_patience'], min_delta=1E-6)

    print('EPOCH    LR        TRAIN     VALID         MSE:T/V              RL:T/V         %NR_t/%NR_v')
    print('=============================================================================================')
    for epoch in range(1, training_params['max_epochs']+1):
        model.train()
        for X_batch, T_batch, y_batch in loader:
            y_pred, g_pred = model(X_batch, T_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, T, y in DataLoader(tDs, shuffle=False, batch_size=tDs.__len__()):
                y_pred_train, g_pred = model(X, T)
                loss_values.append(loss_fn(y_pred_train,y).item())
                mse_t = losses.mseLoss(y_pred_train,y).item()  
                rl_t = losses.realizabilityLoss(y_pred_train).item()  
            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__()):
                y_pred_val, g_pred = model(X, T)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_v = losses.mseLoss(y_pred_val,y).item()  
                rl_v = losses.realizabilityLoss(y_pred_val).item() 

        if val_loss_values[-1] < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)
        if early_stopper.early_stop(val_loss_values[-1]):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_t:.5f} / {mse_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")
            break   

        if (epoch % 10==0 or epoch==1):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_t:.5f} / {mse_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")

        lr_scheduler.step()
    return best_model, loss_values, val_loss_values
"""
"""
def early_stopped_mlp_training_run(model_params, training_params, df_tv):
    loss_fn = bLoss
    loss_values = []
    val_loss_values = []
    df_valid = df_tv[df_tv['Case'].isin(training_params['val_set'])]
    df_train = df_tv[~df_tv['Case'].isin(training_params['val_set'])]
    tDs = bDataset(df_train, input_features=model_params['input_features'])
    vDs = bDataset(df_valid, input_features=model_params['input_features'], scaler = tDs.X_scaler)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])

    print(f'Training points: {len(df_train)}, validation points {len(df_valid)}')

    model = MLP(input_dim = len(model_params['input_features']),
                 n_hidden = model_params['n_hidden'],
                 neurons = model_params['neurons'],
                 activation_function = model_params['activation_function']
                ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=training_params['learning_rate_decay'])
    early_stopper = EarlyStopper(patience=training_params['early_stopping_patience'], min_delta=1E-6)

    print('EPOCH    LR        TRAIN     VALID         MSE:T/V              RL:T/V         %NR_t/%NR_v')
    print('=============================================================================================')
    for epoch in range(1, training_params['max_epochs']+1):
        model.train()
        for X_batch, T_batch, y_batch in loader:
            y_pred = model(X_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for X, T, y in DataLoader(tDs, shuffle=False, batch_size=tDs.__len__()):
                y_pred_train = model(X)
                loss_values.append(loss_fn(y_pred_train,y).item())
                mse_t = mseLoss(y_pred_train,y).item()  
                rl_t = realizabilityLoss(y_pred_train).item()  
            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__()):
                y_pred_val = model(X)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_v = mseLoss(y_pred_val,y).item()  
                rl_v = realizabilityLoss(y_pred_val).item() 

        if val_loss_values[-1] < early_stopper.min_validation_loss:
            best_model = copy.deepcopy(model)
        if early_stopper.early_stop(val_loss_values[-1]):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_t:.5f} / {mse_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")
            break   

        if (epoch % 10==0 or epoch==1):
            print(f"{epoch:3d}   "
                  f"{lr_scheduler._last_lr[-1]:.3e}   "
                  f"{loss_values[-1]:.5f}   "
                  f"{val_loss_values[-1]:.5f}   "
                  f"{mse_t:.5f} / {mse_v:.5f}   "
                  f"{rl_t:.5f} / {rl_v:.5f}   "
                  f"{count_nonrealizable(y_pred_train)/len(y_pred_train)*100:.2f}% / {count_nonrealizable(y_pred_val)/len(y_pred_val)*100:.2f}%")

        lr_scheduler.step()
    return best_model, loss_values, val_loss_values
"""