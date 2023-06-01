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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
import copy

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

def bLoss(outputs, labels, alpha = 1):
    #specifying the batch size
    #if torch.nonzero(torch.isnan(outputs)).sum().item() != 0: print(f'NaNs: {torch.nonzero(torch.isnan(outputs)).sum().item()}')
    #if torch.nonzero(torch.isinf(outputs)).sum().item() != 0: print(f'infs: {torch.nonzero(torch.isinf(outputs)).sum().item()}')
    outputs = torch.nan_to_num(outputs)
    batch_size = outputs.size()[0]
    se = ((outputs[:,0,0] - labels[:,0,0])**2 \
           + (outputs[:,0,1] - labels[:,0,1])**2 \
           + (outputs[:,0,2] - labels[:,0,2])**2 \
           + (outputs[:,1,1] - labels[:,1,1])**2 \
           + (outputs[:,1,2] - labels[:,1,2])**2 \
           + (outputs[:,2,2] - labels[:,2,2])**2 \
          )/6
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero = torch.zeros_like(outputs)
    zero_eig = torch.zeros_like(eigs[:,0])
    re = (torch.maximum(torch.maximum(outputs[:,0,0]-2/3, -(outputs[:,0,0] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,1]-2/3, -(outputs[:,1,1] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,2,2]-2/3, -(outputs[:,2,2] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,1]-1/2, -(outputs[:,0,1] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,2]-1/2, -(outputs[:,0,2] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,2]-1/2, -(outputs[:,1,2] + 1/2)), zero[:,0,0])**2 \
          )/6 \
        + (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2) \
            + (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    return (se+alpha*re).mean()

def mseLoss(outputs, labels):
    #specifying the batch size
    batch_size = outputs.size()[0]
    se = ((outputs[:,0,0] - labels[:,0,0])**2 \
           + (outputs[:,0,1] - labels[:,0,1])**2 \
           + (outputs[:,0,2] - labels[:,0,2])**2 \
           + (outputs[:,1,1] - labels[:,1,1])**2 \
           + (outputs[:,1,2] - labels[:,1,2])**2 \
           + (outputs[:,2,2] - labels[:,2,2])**2 \
          )/6
    return (se).mean()

def realizLoss_components(outputs, labels, alpha = 1):
    #specifying the batch size
    outputs = torch.nan_to_num(outputs)
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero = torch.zeros_like(outputs)
    re = (torch.maximum(torch.maximum(outputs[:,0,0]-2/3, -(outputs[:,0,0] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,1]-2/3, -(outputs[:,1,1] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,2,2]-2/3, -(outputs[:,2,2] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,1]-1/2, -(outputs[:,0,1] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,2]-1/2, -(outputs[:,0,2] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,2]-1/2, -(outputs[:,1,2] + 1/2)), zero[:,0,0])**2 \
          )/6 
    return (alpha*re).mean()

def realizLoss_eig(outputs, labels, alpha = 1):
    #specifying the batch size
    outputs = torch.nan_to_num(outputs)
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero = torch.zeros_like(outputs)
    zero_eig = torch.zeros_like(eigs[:,0])
    re = (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2) \
            + (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    return (alpha*re).mean()

def realizLoss_eig1(outputs, labels, alpha = 1):
    #specifying the batch size
    outputs = torch.nan_to_num(outputs)
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero = torch.zeros_like(outputs)
    zero_eig = torch.zeros_like(eigs[:,0])
    re = (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2) 
    return (alpha*re).mean()

def realizLoss_eig2(outputs, labels, alpha = 1):
    #specifying the batch size
    outputs = torch.nan_to_num(outputs)
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero = torch.zeros_like(outputs)
    zero_eig = torch.zeros_like(eigs[:,0])
    re = (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    print(eigs)
    return (alpha*re).mean()

def realizLoss(outputs, labels, alpha = 1):
    #specifying the batch size
    outputs = torch.nan_to_num(outputs)
    batch_size = outputs.size()[0]
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero = torch.zeros_like(outputs)
    zero_eig = torch.zeros_like(eigs[:,0])
    re = (torch.maximum(torch.maximum(outputs[:,0,0]-2/3, -(outputs[:,0,0] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,1]-2/3, -(outputs[:,1,1] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,2,2]-2/3, -(outputs[:,2,2] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,1]-1/2, -(outputs[:,0,1] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,2]-1/2, -(outputs[:,0,2] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,2]-1/2, -(outputs[:,1,2] + 1/2)), zero[:,0,0])**2 \
          )/6 \
        + (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2) \
            + (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    return (alpha*re).mean()

class TBNN(nn.Module):
    def __init__(self, N: int, input_dim: int, n_hidden: int, neurons: int, activation_function):
        super().__init__()
        self.N = N
        self.input_dim = input_dim   
        
        self.gn = nn.Linear(neurons,self.N)
        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
                    
    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        b_pred = torch.sum(gn.view(-1,self.N,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
    

class MLP(nn.Module):
    def __init__(self, input_dim: int, n_hidden: int, neurons: int, activation_function):
        super().__init__()
        self.input_dim = input_dim   

        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
        self.output = nn.Linear(neurons, 5)
                    
    def forward(self, x):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        output = self.output(x)
        #b_pred = self.output(x).view(-1,3,3)
        b_pred = torch.column_stack((output[:,0],output[:,1],output[:,2],output[:,1],output[:,3],output[:,4],output[:,2],output[:,4],-output[:,0]-output[:,3])).view(-1,3,3)
        return b_pred
    
class bDataset(Dataset):
    def __init__(self, df, input_features, scaler=None):
        # convert into PyTorch tensors and remember them
        if scaler == None:
            self.X_scaler = StandardScaler()
            self.X_scaler.fit((df[input_features].values.astype(np.float32)))
        else: 
            self.X_scaler = scaler
        self.X = torch.from_numpy( self.X_scaler.transform(df[input_features].values.astype(np.float32))).to(device)
        self.y = torch.from_numpy( np.float32(self.assemble_b(df))).to(device)
        self.T = torch.from_numpy( np.float32(self.assemble_T(df))).to(device)
        
    def assemble_T(self,df):
        T = np.empty((self.__len__(),10,3,3))
        for i in range(10):
            T[:,i,0,0] = df[f'komegasst_T{i+1}_11']
            T[:,i,0,1] = df[f'komegasst_T{i+1}_12']
            T[:,i,0,2] = df[f'komegasst_T{i+1}_13']
            T[:,i,1,1] = df[f'komegasst_T{i+1}_22']
            T[:,i,1,2] = df[f'komegasst_T{i+1}_23']
            T[:,i,2,2] = df[f'komegasst_T{i+1}_22']
            
            T[:,i,1,0] = T[:,i,0,1]
            T[:,i,2,0] = T[:,i,0,2]
            T[:,i,2,1] = T[:,i,1,2]
        return T

    def assemble_b(self,df):
        b = np.empty((self.__len__(),3,3))
        b[:,0,0] = df['DNS_b_11']
        b[:,0,1] = df['DNS_b_12']
        b[:,0,2] = df['DNS_b_13']
        b[:,1,1] = df['DNS_b_22']
        b[:,1,2] = df['DNS_b_23']
        b[:,2,2] = df['DNS_b_33']
        b[:,1,0] = b[:,0,1]
        b[:,2,0] = b[:,0,2]
        b[:,2,1] = b[:,1,2]
        return b
        
    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        Tn = self.T[idx]
        return features, Tn, target
    
def count_nonrealizable(b):
    b = torch.nan_to_num(b)
    eigs = torch.sort(torch.real(torch.linalg.eigvals(b)),descending=True)[0]
    zero = torch.zeros_like(b)
    zero_eig = torch.zeros_like(eigs[:,0])
    re = (torch.maximum(torch.maximum(b[:,0,0]-2/3, -(b[:,0,0] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,1,1]-2/3, -(b[:,1,1] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,2,2]-2/3, -(b[:,2,2] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,0,1]-1/2, -(b[:,0,1] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,0,2]-1/2, -(b[:,0,2] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,1,2]-1/2, -(b[:,1,2] + 1/2)), zero[:,0,0])**2 \
          )/6 \
        + (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2) \
            + (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    n_nr = torch.count_nonzero(re)
    return n_nr

class Result(defaultdict):
    def __init__(self, value=None):
        super(results, self).__init__(results)
        self.value = value


def early_stopped_tbnn_training_run(model_params, training_params, df_tv):
    loss_fn = bLoss
    loss_values = []
    val_loss_values = []
    df_valid = df_tv[df_tv['Case'].isin(training_params['val_set'])]
    df_train = df_tv[~df_tv['Case'].isin(training_params['val_set'])]
    tDs = bDataset(df_train, input_features=model_params['input_features'])
    vDs = bDataset(df_valid, input_features=model_params['input_features'], scaler = tDs.X_scaler)
    loader = DataLoader(tDs, shuffle=True, batch_size=training_params['batch_size'])

    print(f'Training points: {len(df_train)}, validation points {len(df_valid)}')

    model = TBNN(N = 10,
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
                mse_t = mseLoss(y_pred_train,y).item()  
                rl_t = realizLoss(y_pred_train,y).item()  
            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__()):
                y_pred_val, g_pred = model(X, T)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_v = mseLoss(y_pred_val,y).item()  
                rl_v = realizLoss(y_pred_val,y).item() 

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
                rl_t = realizLoss(y_pred_train,y).item()  
            for X, T, y in DataLoader(vDs, shuffle=False, batch_size=vDs.__len__()):
                y_pred_val = model(X)
                val_loss_values.append(loss_fn(y_pred_val,y).item())   
                mse_v = mseLoss(y_pred_val,y).item()  
                rl_v = realizLoss(y_pred_val,y).item() 

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