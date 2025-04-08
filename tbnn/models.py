import torch
import torch.nn as nn
import pickle
from tbnn.barcode import datestamp
from kan import KAN

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
    
class TBNNiv(nn.Module):
    """
    TBNNPerp, with g1 forced to be -1, and k correction factor
    """
    def __init__(self, N: int, input_dim: int, n_hidden: int, neurons: int, activation_function, input_feature_names: list):
        super().__init__()
        self.N = N
        self.input_dim = input_dim   
        self.gn = nn.Linear(neurons,self.N) #N-1 for 9 g's, N for 9 g's + k correction
        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
        self.input_feature_names = input_feature_names
        self.input_feature_scaler = None
        self.barcode = f'TBNNiv-{datestamp}'

                    
    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        gn = torch.cat((-torch.ones_like(gn[:,0]).view(-1,1), gn), 1)
        b_pred = torch.sum(gn[:,0:-1].view(-1,self.N,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
    
class KCNN(nn.Module):
    """
    KCNN - corrects TKE
    """
    def __init__(self, input_dim: int, n_hidden: int, neurons: int, activation_function, input_feature_names: list):
        super().__init__() # does super make a difference here?
        self.input_dim = input_dim   
        self.output = nn.Linear(neurons,1)
        self.activation_function = activation_function
        self.hidden = nn.ModuleList()
        for k in range(n_hidden):
            self.hidden.append(nn.Linear(input_dim, neurons))
            input_dim = neurons  # For the next layer
        self.input_feature_names = input_feature_names
        self.input_feature_scaler = None
        self.barcode = f'KCNN-{datestamp}'
     
    def forward(self, x):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        Delta = self.output(x)
        return Delta,
    
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
        
        
class kanTBNN(KAN):
    """
    Implementing basic pykan formulation for TBNN
    """
    
    def __init__(self, width, grid, k, seed, input_feature_names: list):
            super().__init__(width, grid, k, seed) 
            self.input_feature_names = input_feature_names
            self.input_feature_scaler = None
            self.barcode = f'kanTBNN-{datestamp}'

    def forward(self, x, Tn):
        gn = super().forward(x)
        b_pred = torch.sum(gn.view(-1,10,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
    
    # Save the model using pickle in a similar way to other TBNN models
    def save_model_as_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
  
    # Optionally save using torch (similar to TBNN models saved by torch.save)
    def save_model_as_torch(self, path):
        torch.save(self.state_dict(), path)

"""
# First need to clone the github repo of ChebyKAN installed 
# from ChebyKANLayer import ChebyKANLayer

class chebyTBNN(ChebyKANLayer):
    
    #Implementing basic cheby formulation for TBNN
    
    def __init__(self, width, grid, k, seed, input_feature_names: list):
            super().__init__(width, grid, k, seed) 
            self.input_feature_names = input_feature_names
            self.input_feature_scaler = None
            self.barcode = f'chebykanTBNN-{datestamp}'

    def forward(self, x, Tn):
        gn = super().forward(x)
        b_pred = torch.sum(gn.view(-1,10,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
    
    # Save the model using pickle in a similar way to other TBNN models
    def save_model_as_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
  
    # Optionally save using torch (similar to TBNN models saved by torch.save)
    def save_model_as_torch(self, path):
        torch.save(self.state_dict(), path)

"""