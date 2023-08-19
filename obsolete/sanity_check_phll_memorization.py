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

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv',
                  'Cases': ['case_1p2'],
                }
#dataset_params['Cases'] = ['case_1p0']

training_params = { 'early_stopping_patience': 100,
                    'max_epochs': 2000,
                    'learning_rate': 0.0001,
                    'learning_rate_decay': 1.0,
                    'batch_size': 32,
                    'val_set': ['case_1p2'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                    'test_set': ['case_1p2']
                }

model_params = {'neurons': 40, 'n_hidden': 3, 'activation_function': nn.SiLU(),                 
                'input_features':[
'komegasst_q6',
'komegasst_q5',
'komegasst_q8',
'komegasst_I1_16',
'komegasst_I1_7',
'komegasst_I1_3',
'komegasst_I2_6',
'komegasst_q3',
'komegasst_I2_3',
'komegasst_I1_4',
'komegasst_I2_7',
'komegasst_I1_35',
'komegasst_q2',
'komegasst_q7',
'komegasst_I1_1',
'komegasst_q4',
'komegasst_I2_8',

#'komegasst_q6',
]
}



df = pd.read_csv(dataset_params['file'])

df = df[df['Case'].isin(dataset_params['Cases'])]
df_test = df
df_tv = df
#df_test = df
#df_tv = df.sample(n=1000)
#df_tv = df


print(f'Dataset: {len(df)}, test: {len(df_test)}, tv: {len(df_tv)}')

model = models.TBNNiii(N = 10,
                input_dim = len(model_params['input_features']),
                n_hidden = model_params['n_hidden'],
                neurons = model_params['neurons'],
                activation_function = model_params['activation_function'],
                input_feature_names=model_params['input_features']
            ).to(device)
sys.stdout = open(f'models/phll_memorization/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.log','w')

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
fig.savefig(f'models/phll_memorization/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.png',dpi=300)

import tbnn.training_utils as training_utils
import numpy as np
from torch.utils.data import Dataset, DataLoader

#df_train = df[~df['Case'].isin(training_params['test_set']+training_params['val_set'])]
#df_val = df[df['Case'].isin(training_params['val_set'])]
df_train = df_tv
df_val = df_tv
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

    nut_L = df_test['komegasst_nut']*-gn[:,0].detach().numpy()
    gn2 = gn[:,1:]
    T = inputs[1][:,1:,:]
    b_perp_pred = torch.sum(gn2.view(-1,9,1,1)*torch.ones_like(T)*T,axis=1).detach().numpy()
    a_perp_pred = b_perp_pred*2*df_test['DNS_k'].to_numpy()[:,None,None]
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

fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,3))
for i, scalar in enumerate(['b_11','b_12','b_13','b_22','b_23','b_33']):
    vmin = min(df_test[f'DNS_{scalar}'])
    vmax = max(df_test[f'DNS_{scalar}'])
    axs[0,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'DNS_{scalar}'])
    axs[1,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'pred_{scalar}_all'])
    axs[0,i].set_aspect(1)
    axs[1,i].set_aspect(1)
fig.savefig(f'models/phll_memorization/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}_b_test.png',dpi=300)

fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,3))
for i, scalar in enumerate(['a_11','a_12','a_13','a_22','a_23','a_33']):
    vmin = min(df_test[f'DNS_{scalar}'])
    vmax = max(df_test[f'DNS_{scalar}'])
    axs[0,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'DNS_{scalar}'])
    axs[1,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'pred_{scalar}_all'])
    axs[0,i].set_aspect(1)
    axs[1,i].set_aspect(1)


fig.savefig(f'models/phll_memorization/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}_a_test.png',dpi=300)


import sys
sys.path.insert(1, '/home/ryley/WDK/ML/code/data_scripting/')
import dataFoam

import numpy as np
from dataFoam.utilities.foamIO.writeFoam_PHLL import writeFoam_ap_PHLL, writeFoam_nut_L_PHLL, writeFoam_genericscalar_PHLL
from dataFoam.utilities.foamIO.readFoam import get_endtime
import os

endtime = '0'
foamdir = '/home/ryley/WDK/ML/scratch/injection/case_1p2_memorization'
#writeFoam_nut_L_PHLL(os.path.join(foamdir,endtime,'nut_L'),nut_L)
writeFoam_ap_PHLL(os.path.join(foamdir,endtime,'aperp'),a_perp_pred)

torch.save(model.state_dict(), f'models/phll_memorization/phll_memorization.state_dict')

with open(f"models/phll_memorization/params_phll_memorization.pickle", 'wb') as handle:
    pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
