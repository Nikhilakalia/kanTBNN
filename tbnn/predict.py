import tbnn.dataloaders as dataloaders
import tbnn.models as models
import torch
from torch.utils.data import DataLoader
import numpy as np

def TBNN(model,df,k_name=None):
    if isinstance(model,models.clusterTBNN):
        return cluster_TBNN(model, df, k_name)
    else:
        return single_TBNN(model,df, k_name)

def cluster_TBNN(model, df):
    #splitr_scaler = model.splitr_scaler
    p = model.splitr.predict_proba(model.splitr_scaler.transform(df[model.splitr_input_features]))
    b_pred = torch.zeros((len(df),3,3))
    gn = torch.zeros((len(df),10))
    for i,tbnni in enumerate(model.tbnn_list):
        ds = dataloaders.aDataset(df, input_features=tbnni.input_feature_names, scaler_X = tbnni.input_feature_scaler, assemble_labels=False)
        #for inputs, labels in DataLoader(ds , shuffle=False, batch_size=ds.__len__()):
        b_predi, gni = tbnni(ds.X,ds.T) #b
        b_pred += torch.from_numpy(p[:,i]).view(-1,1,1)*b_predi
        gn += torch.from_numpy(p[:,i]).view(-1,1)*gni

    gn2 = gn[:,1:-1] #non-linear gn
    T = ds.T[:,1:,:] #non-linear T
    b_perp_pred = torch.sum(gn2.view(-1,9,1,1)*torch.ones_like(T)*T,axis=1).detach().numpy() #non-linear b

    return b_pred, gn, b_perp_pred

def single_TBNN(model, df, k_name = None):
    ds = dataloaders.aDataset(df, input_features=model.input_feature_names, scaler_X = model.input_feature_scaler, assemble_labels=False)
    b_pred, gn = model(ds.X,ds.T) #b
    gn2 = gn[:,1:-1] #non-linear gn
    T = ds.T[:,1:,:] #non-linear T
    b_perp_pred = torch.sum(gn2.view(-1,9,1,1)*torch.ones_like(T)*T,axis=1) #non-linear b

    if k_name is not None:
        a_pred = 2*((torch.exp(gn[:,-1]).detach().numpy())*df[k_name].to_numpy())[:,None,None]*b_pred.detach().numpy()
        a_perp_pred = 2*((torch.exp(gn[:,-1]).detach().numpy())*df[k_name].to_numpy())[:,None,None]*b_perp_pred.detach().numpy()
    else: 
        a_pred = None
        a_perp_pred = None
    # gn, b_pred, b_perp_pred returned as torch tensors, a_pred, a_perp_pred returned as numpy arrays
    return gn, b_pred, b_perp_pred, a_pred, a_perp_pred 