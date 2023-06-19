from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import tbnn.devices as devices
device = devices.get_device()
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class bDataset(Dataset):
    def __init__(self, df, input_features, scaler_X=None):
        if scaler_X == None:
            self.scaler_X = StandardScaler()
            self.scaler_X.fit((df[input_features].values.astype(np.float32)))
        else: 
            self.scaler_X = scaler_X
        self.X = torch.from_numpy( self.scaler_X.transform(df[input_features].values.astype(np.float32))).to(device)
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
            T[:,i,2,2] = df[f'komegasst_T{i+1}_33']
            
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
        return len(self.X)
 
    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        Tn = self.T[idx]
        return features, Tn, target
    
class bDatasetPlus(bDataset):
    def __init__(self, df, input_features, scaler_X=None, scaler_g1tilde=None):
        super().__init__(df, input_features, scaler_X)
        if scaler_g1tilde == None:
            self.scaler_g1tilde = MinMaxScaler()
            self.scaler_g1tilde.fit((df['DNS_g1tilde'].values.astype(np.float32).reshape(-1, 1)))
        else: 
            self.scaler_g1tilde = scaler_g1tilde
        self.g1tilde = torch.from_numpy(self.scaler_g1tilde.transform(df['DNS_g1tilde'].values.astype(np.float32).reshape(-1, 1))).to(device)

    def __getitem__(self, idx):
        # Note: this may need to be changed, pay attention to whether this works
        features, Tn, target = super().__getitem__(idx)
        g1tilde = self.g1tilde[idx]
        return features, Tn, target, g1tilde


    