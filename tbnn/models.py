import torch
import torch.nn as nn
import pickle
from tbnn.barcode import datestamp

class TBNNi(nn.Module):
    """
    Plain TBNN.
    """
    def __init__(self, N: int, input_dim: int, n_hidden: int, neurons: int, activation_function, input_feature_names: list):
        super().__init__()
        self.N = N
        self.input_dim = input_dim   
        self.gn = nn.Linear(neurons,self.N)
        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
        self.input_feature_names = input_feature_names
        self.input_feature_scaler = None
        self.barcode = f'TBNNi-{datestamp}'


    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        b_pred = torch.sum(gn.view(-1,self.N,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
       
class TBNNii(nn.Module):
    """
    TBNN+, with g1 forced to be negative.
    """
    def __init__(self, N: int, input_dim: int, n_hidden: int, neurons: int, activation_function, input_feature_names: list):
        super().__init__()
        self.N = N
        self.input_dim = input_dim   
        self.gn = nn.Linear(neurons,self.N)
        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
        self.input_feature_names = input_feature_names
        self.input_feature_scaler = None
        self.barcode = f'TBNNii-{datestamp}'

                    
    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        gn[:,0] = -torch.exp(gn[:,0])
        b_pred = torch.sum(gn.view(-1,self.N,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn

class TBNNiii(nn.Module):
    """
    TBNNPerp, with g1 forced to be -1.
    """
    def __init__(self, N: int, input_dim: int, n_hidden: int, neurons: int, activation_function, input_feature_names: list):
        super().__init__()
        self.N = N
        self.input_dim = input_dim   
        self.gn = nn.Linear(neurons,self.N-1)
        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
        self.input_feature_names = input_feature_names
        self.input_feature_scaler = None
        self.barcode = f'TBNNiii-{datestamp}'

                    
    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        gn = torch.cat((-torch.ones_like(gn[:,0]).view(-1,1), gn), 1)
        b_pred = torch.sum(gn.view(-1,self.N,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
    
class clusterTBNN():
    """
    An assembly of models.
    """
    def __init__(self, model_dict):
        self.model_dict = model_dict
        self.assemble_models()
        self.splitr_input_features = model_dict['splitr_input_features']
        self.barcode = f'clusterTBNN-{datestamp}'

    def assemble_models(self):
        self.splitr = pickle.load(open(self.model_dict['splitr'], 'rb')) 
        self.splitr_scaler = pickle.load(open(self.model_dict['splitr_scaler'], 'rb')) 
        self.tbnn_list = []
        for tbnni in self.model_dict['models'].keys():
            self.tbnn_list.append(torch.load(self.model_dict['models'][tbnni]))
        
        
