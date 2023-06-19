import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TBNN(nn.Module):
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
                    
    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        b_pred = torch.sum(gn.view(-1,self.N,1,1)*torch.ones_like(Tn)*Tn,axis=1)
        return b_pred, gn
    
class TBNNPlus(nn.Module):
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
                    
    def forward(self, x, Tn):
        for layer in self.hidden:
            x = self.activation_function(layer(x))
        gn = self.gn(x)
        gn[:,0] = -torch.exp(gn[:,0])
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
        b_pred = torch.column_stack((output[:,0],output[:,1],output[:,2],output[:,1],output[:,3],output[:,4],output[:,2],output[:,4],-output[:,0]-output[:,3])).view(-1,3,3)
        return b_pred