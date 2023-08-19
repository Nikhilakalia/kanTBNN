import pandas as pd
import torch.nn as nn
from tbnn.training_utils import early_stopped_tbnn_training_run
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import tbnn.models as models
import tbnn.devices as devices
import tbnn.dataloaders as dataloaders
import tbnn.losses as losses
import sys
device = devices.get_device()

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean_split.csv',
                  'Cases': ['fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'],
                }

training_params = { 'early_stopping_patience': 500,
                    'max_epochs': 2000,
                    'learning_rate': 0.002,
                    'learning_rate_decay': 0.999,
                    'batch_size': 32,
                    'val_set': ['fp_3030','fp_1410','fp_4060'],
                    'test_set': ['fp_3630']
                }

model_params = {'neurons': 20, 'n_hidden': 5, 'activation_function': nn.SiLU(),                 
                'input_features':['komegasst_I1_1',
'komegasst_I1_3',
'komegasst_I1_4',
'komegasst_I1_5',
'komegasst_I1_16',
#'komegasst_I1_7',
#'komegasst_I1_9',
#'komegasst_I1_10',
#'komegasst_I1_12',
#'komegasst_I1_13',
#'komegasst_I1_16',
'komegasst_q5',
#'komegasst_q6',
]
}



df = pd.read_csv(dataset_params['file'])

df = df[df['Case'].isin(dataset_params['Cases'])]

df_test = df[df['Case'].isin(training_params['test_set'])]
df_tv = df[~df['Case'].isin(training_params['test_set'])]
print(f'Dataset: {len(df)}, test: {len(df_test)}, tv: {len(df_tv)}')

model = models.TBNNiii(N = 10,
                input_dim = len(model_params['input_features']),
                n_hidden = model_params['n_hidden'],
                neurons = model_params['neurons'],
                activation_function = model_params['activation_function'],
                input_feature_names=model_params['input_features']
            ).to(device)

sys.stdout = open(f'models/fp_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.log','w')

model, loss_vals, val_loss_vals  = early_stopped_tbnn_training_run(model = model,
                                                                   training_params = training_params,
                                                                   df_tv = df_tv,
                                                                   data_loader = dataloaders.aDataset, 
                                                                   loss_fn = losses.aLoss)

fig, ax = plt.subplots(1,figsize=(5,5))
ax.plot(loss_vals,'-',color='b')
ax.plot(val_loss_vals,'--',color='r')
ax.semilogy()
fig.tight_layout()
fig.savefig(f'models/fp_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.png',dpi=300)

import tbnn.training_utils as training_utils
import numpy as np
from torch.utils.data import Dataset, DataLoader

df_train = df[~df['Case'].isin(training_params['test_set']+training_params['val_set'])]
df_val = df[df['Case'].isin(training_params['val_set'])]

tDs = dataloaders.aDataset(df_train, input_features=model_params['input_features'])
vDs = dataloaders.aDataset(df_val, input_features=model_params['input_features'],scaler_X = tDs.scaler_X)
testDs = dataloaders.aDataset(df_test, input_features=model_params['input_features'],scaler_X = tDs.scaler_X)

model.eval()
for inputs,labels_train in DataLoader(tDs , shuffle=False, batch_size=testDs.__len__()):
    y_pred_train, gn = model(*inputs)

for inputs,labels_val in DataLoader(vDs , shuffle=False, batch_size=testDs.__len__()):
    y_pred_val, gn = model(*inputs)

for inputs,labels_test in DataLoader(testDs, shuffle=False, batch_size=testDs.__len__()):
    y_pred_test, gn = model(*inputs)
    #print(f"loss: {losses.bLoss(y_pred_test,y)}")
    #print(f"loss: {losses.mseLoss(y_pred_test,y)}")
    #print(f"loss: {losses.realizabilityPenalty(y)}")
    #print(f"loss: {losses.realizabilityPenalty_components(y)}")
    #print(f"loss: {losses.realizabilityPenalty_eigs(y)}")


    #val_loss_values.append(loss_fn(y_pred_val,y).item())   
    #mse_v = mseLoss(y_pred_val,y).item()  
    #rl_v = realizLoss(y_pred_val,y).item()

print(f'Training losses: {losses.aLoss(y_pred_train, *labels_train)}')
print(f'Validation losses: {losses.aLoss(y_pred_val, *labels_val)}')
print(f'Test losses: {losses.aLoss(y_pred_test, *labels_test)}')

df_test[f'pred_b_11_all'] = y_pred_test.detach().numpy()[:,0,0]
df_test[f'pred_b_12_all'] = y_pred_test.detach().numpy()[:,0,1]
df_test[f'pred_b_13_all'] = y_pred_test.detach().numpy()[:,0,2]
df_test[f'pred_b_22_all'] = y_pred_test.detach().numpy()[:,1,1]
df_test[f'pred_b_23_all'] = y_pred_test.detach().numpy()[:,1,2]
df_test[f'pred_b_33_all'] = y_pred_test.detach().numpy()[:,2,2]

df_test[f'pred_a_11_all'] = y_pred_test.detach().numpy()[:,0,0]*2*df_test[f'DNS_k']
df_test[f'pred_a_12_all'] = y_pred_test.detach().numpy()[:,0,1]*2*df_test[f'DNS_k']
df_test[f'pred_a_13_all'] = y_pred_test.detach().numpy()[:,0,2]*2*df_test[f'DNS_k']
df_test[f'pred_a_22_all'] = y_pred_test.detach().numpy()[:,1,1]*2*df_test[f'DNS_k']
df_test[f'pred_a_23_all'] = y_pred_test.detach().numpy()[:,1,2]*2*df_test[f'DNS_k']
df_test[f'pred_a_33_all'] = y_pred_test.detach().numpy()[:,2,2]*2*df_test[f'DNS_k']

df_test[f'pred_g1'] = gn[:,0].detach().numpy()
df_test[f'pred_g2'] = gn[:,1].detach().numpy()
df_test[f'pred_g3'] = gn[:,2].detach().numpy()
df_test[f'pred_g4'] = gn[:,3].detach().numpy()
df_test[f'pred_g5'] = gn[:,4].detach().numpy()
df_test[f'pred_g6'] = gn[:,5].detach().numpy()
df_test[f'pred_g7'] = gn[:,6].detach().numpy()
df_test[f'pred_g8'] = gn[:,7].detach().numpy()
df_test[f'pred_g9'] = gn[:,8].detach().numpy()
df_test[f'pred_g10'] = gn[:,9].detach().numpy()

fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(15,15))
axs[0,0].scatter(df_test['komegasst_C_2'],df_test['DNS_b_11'])
axs[0,0].scatter(df_test['komegasst_C_2'],df_test['pred_b_11_all'])

axs[0,1].scatter(df_test['komegasst_C_2'],df_test['DNS_b_12'])
axs[0,1].scatter(df_test['komegasst_C_2'],df_test['pred_b_12_all'])

axs[0,2].scatter(df_test['komegasst_C_2'],df_test['DNS_b_13'])
axs[0,2].scatter(df_test['komegasst_C_2'],df_test['pred_b_13_all'])

axs[1,1].scatter(df_test['komegasst_C_2'],df_test['DNS_b_22'])
axs[1,1].scatter(df_test['komegasst_C_2'],df_test['pred_b_22_all'])

axs[1,2].scatter(df_test['komegasst_C_2'],df_test['DNS_b_23'])
axs[1,2].scatter(df_test['komegasst_C_2'],df_test['pred_b_23_all'])

axs[2,2].scatter(df_test['komegasst_C_2'],df_test['DNS_b_33'])
axs[2,2].scatter(df_test['komegasst_C_2'],df_test['pred_b_33_all'])

fig.savefig(f'models/fp_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}_b_test.png',dpi=300)

fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(15,15))
axs[0,0].scatter(df_test['komegasst_C_2'],df_test['DNS_a_11'])
axs[0,0].scatter(df_test['komegasst_C_2'],df_test['pred_a_11_all'])

axs[0,1].scatter(df_test['komegasst_C_2'],df_test['DNS_a_12'])
axs[0,1].scatter(df_test['komegasst_C_2'],df_test['pred_a_12_all'])

axs[0,2].scatter(df_test['komegasst_C_2'],df_test['DNS_a_13'])
axs[0,2].scatter(df_test['komegasst_C_2'],df_test['pred_a_13_all'])

axs[1,1].scatter(df_test['komegasst_C_2'],df_test['DNS_a_22'])
axs[1,1].scatter(df_test['komegasst_C_2'],df_test['pred_a_22_all'])

axs[1,2].scatter(df_test['komegasst_C_2'],df_test['DNS_a_23'])
axs[1,2].scatter(df_test['komegasst_C_2'],df_test['pred_a_23_all'])

axs[2,2].scatter(df_test['komegasst_C_2'],df_test['DNS_a_33'])
axs[2,2].scatter(df_test['komegasst_C_2'],df_test['pred_a_33_all'])

fig.savefig(f'models/fp_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}_a_test.png',dpi=300)

