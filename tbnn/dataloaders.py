from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import tbnn.devices as devices
device = devices.get_device()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#print(device) #sanity check if i am on gpu or not 

class bDataset(Dataset):
    def __init__(self, df, input_features, scaler_X=None, Perp=True, assemble_labels=True):
        self.Perp = Perp
        if input_features is not None:
            if scaler_X == None:
                self.scaler_X = StandardScaler()
                self.scaler_X.fit((df[input_features].values.astype(np.float32)))
            else: 
                self.scaler_X = scaler_X
            self.X = torch.from_numpy(self.scaler_X.transform(df[input_features].values.astype(np.float32))).to(device)
            self.T = torch.from_numpy(np.float32(self.assemble_T(df))).to(device)
        else:
            self.X = torch.empty(len(df))
            self.T = torch.empty(len(df))

        if assemble_labels:
            self.b = torch.from_numpy(np.float32(self.assemble_b(df))).to(device)
        
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

        if self.Perp:
            T[:,0,0,0] = df[f'DNS_S_11']
            T[:,0,0,1] = df[f'DNS_S_12']
            T[:,0,0,2] = df[f'DNS_S_13']
            T[:,0,1,1] = df[f'DNS_S_22']
            T[:,0,1,2] = df[f'DNS_S_23']
            T[:,0,2,2] = df[f'DNS_S_33']
            T[:,0,1,0] = T[:,0,0,1]
            T[:,0,2,0] = T[:,0,0,2]
            T[:,0,2,1] = T[:,0,1,2]
            T[:,0,:] = T[:,0,:]*(np.divide(df[f'komegasst_nut'].to_numpy()[:,None,None],np.maximum(1E-10,df[f'DNS_k'].to_numpy())[:,None,None]))
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
        X = self.X[idx]
        b = self.b[idx]
        Tn = self.T[idx]
        return (X, Tn), (b,)
        
class aDataset(bDataset):
    def __init__(self, df, input_features, scaler_X=None, Perp=True, assemble_labels=True):
        super().__init__(df, input_features, scaler_X, Perp, assemble_labels)
        if assemble_labels:
            self.k = torch.from_numpy(np.float32(self.assemble_k(df))).to(device)
            self.a = torch.from_numpy(np.float32(self.assemble_a(df))).to(device)
            self.amagmean = torch.from_numpy(np.float32(self.assemble_amagmean(df))).to(device)

    def assemble_k(self,df):
        k = df['DNS_k']
        return k

    def assemble_amagmean(self,df):
        amagmean = df['DNS_amagmean']
        return amagmean 

    
    def assemble_a(self,df):
        a = np.empty((self.__len__(),3,3))
        a[:,0,0] = df['DNS_a_11']
        a[:,0,1] = df['DNS_a_12']
        a[:,0,2] = df['DNS_a_13'] 
        a[:,1,1] = df['DNS_a_22']
        a[:,1,2] = df['DNS_a_23']
        a[:,2,2] = df['DNS_a_33']
        a[:,1,0] = a[:,0,1]
        a[:,2,0] = a[:,0,2]
        a[:,2,1] = a[:,1,2]
        return a
    
    def __getitem__(self, idx):
        X = self.X[idx]
        a = self.a[idx]
        amagmean = self.amagmean[idx]
        k = self.k[idx]
        Tn = self.T[idx]
        return (X, Tn), (k, a, amagmean)
    

class DeltaDataset(bDataset):
    def __init__(self, df, input_features, scaler_X=None, Perp=True, assemble_labels=True, filter=True):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()
        super().__init__(df, input_features, scaler_X, Perp, assemble_labels)

        if assemble_labels:
            self.Delta = torch.from_numpy(np.float32(self.assemble_Delta(df))).to(device)

    def assemble_Delta(self,df):
        Delta = df['Delta']
        return Delta
    
    def __getitem__(self, idx):
        X = self.X[idx]
        Delta = self.Delta[idx]
        return (X, ), (Delta,)
    

class kDataset(bDataset):
    def __init__(self, df, input_features, scaler_X=None, Perp=True, assemble_labels=True):
        super().__init__(df, input_features, scaler_X, Perp, assemble_labels)
        self.k_RANS = torch.from_numpy(np.float32(self.assemble_k_RANS(df))).to(device)
        if assemble_labels:
            self.k_DNS = torch.from_numpy(np.float32(self.assemble_k_DNS(df))).to(device)

    def assemble_k_RANS(self,df):
        k = df['komegasst_k']
        return k
    
    def assemble_k_DNS(self,df):
        k = df['DNS_k']
        return k
    
    def __getitem__(self, idx):
        X = self.X[idx]
        k_RANS = self.k_RANS[idx]
        k_DNS = self.k_DNS[idx]
        return (X, k_RANS), (k_DNS,)
    