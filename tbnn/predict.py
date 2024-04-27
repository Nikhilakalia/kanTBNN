import tbnn.dataloaders as dataloaders
import tbnn.models as models
import torch
from torch.utils.data import DataLoader

def TBNN(model,df):
    if isinstance(model,models.clusterTBNN):
        return cluster_TBNN(model, df)
    else:
        return single_TBNN(model,df)

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

    gn2 = gn[:,1:] #non-linear gn
    T = ds.T[:,1:,:] #non-linear T
    b_perp_pred = torch.sum(gn2.view(-1,9,1,1)*torch.ones_like(T)*T,axis=1).detach().numpy() #non-linear b

    return b_pred, gn, b_perp_pred

def single_TBNN(model, df):
    ds = dataloaders.aDataset(df, input_features=model.input_feature_names, scaler_X = model.input_feature_scaler, assemble_labels=False)
    #for inputs, labels in DataLoader(ds , shuffle=False, batch_size=ds.__len__()):
    b_pred, gn = model(ds.X,ds.T) #b
    gn2 = gn[:,1:] #non-linear gn
    T = ds.T[:,1:,:] #non-linear T
    b_perp_pred = torch.sum(gn2.view(-1,9,1,1)*torch.ones_like(T)*T,axis=1).detach().numpy() #non-linear b
    return b_pred, gn, b_perp_pred