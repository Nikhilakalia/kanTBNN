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
device = devices.get_device()

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean_split.csv',
                  'Cases': ['squareDuctAve_Re_1100',
       'squareDuctAve_Re_1150',
       'squareDuctAve_Re_1250',
       'squareDuctAve_Re_1300',
       'squareDuctAve_Re_1350',
       'squareDuctAve_Re_1400',
       'squareDuctAve_Re_1500',
       'squareDuctAve_Re_1600',
       'squareDuctAve_Re_1800',
       'squareDuctAve_Re_2000',
       'squareDuctAve_Re_2205',
       'squareDuctAve_Re_2400',
       'squareDuctAve_Re_2600',
       'squareDuctAve_Re_2900',
       'squareDuctAve_Re_3200',
       'squareDuctAve_Re_3500'],
                }
dataset_params['Cases'] = ['squareDuctAve_Re_2000']

training_params = { 'early_stopping_patience': 500,
                    'max_epochs': 500,
                    'learning_rate': 0.0001,
                    'learning_rate_decay': 1.0,
                    'batch_size': 32,
                    'val_set': ['squareDuctAve_Re_2000'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                    'test_set': ['squareDuctAve_Re_2000']
                }

model_params = {'neurons': 30, 'n_hidden': 5, 'activation_function': nn.SiLU(),                 
                'input_features':[
#'komegasst_q2',
'komegasst_q5',
'komegasst_q6',
#'komegasst_q8',
'komegasst_I1_1',
'komegasst_I1_4',
'komegasst_I1_11',
'komegasst_I2_4',
'komegasst_I1_21',
'komegasst_I1_10',
'komegasst_I1_15',
'komegasst_I1_16',
'komegasst_I1_19',
'komegasst_I1_25',
'komegasst_I1_44',
'komegasst_I1_43',
#'komegasst_I1_35',
#'komegasst_q3',
#'komegasst_I1_40',
#'komegasst_q7',
#'komegasst_I1_17',
#'komegasst_I1_36',
#'komegasst_I1_27',
#'komegasst_I1_41',
#'komegasst_q6',
]
}



df = pd.read_csv(dataset_params['file'])

df = df[df['Case'].isin(dataset_params['Cases'])]
#df_test = df[df['Case'].isin(training_params['test_set'])]
#df_tv = df[~df['Case'].isin(training_params['test_set'])]
df_test = df
df_tv = df


print(f'Dataset: {len(df)}, test: {len(df_test)}, tv: {len(df_tv)}')

model = models.TBNNiii(N = 10,
                input_dim = len(model_params['input_features']),
                n_hidden = model_params['n_hidden'],
                neurons = model_params['neurons'],
                activation_function = model_params['activation_function'],
                input_feature_names=model_params['input_features']
            ).to(device)

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
fig.savefig(f'models/duct_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}.png',dpi=300)

import tbnn.training_utils as training_utils
import numpy as np
from torch.utils.data import Dataset, DataLoader

#df_train = df[~df['Case'].isin(training_params['test_set']+training_params['val_set'])]
#df_val = df[df['Case'].isin(training_params['val_set'])]
df_train = df
df_val = df
df_test = pd.read_csv('/home/ryley/WDK/ML/dataset/squareDuct_Re_2000.csv')
tDs = dataloaders.aDataset(df_train, input_features=model_params['input_features'])
vDs = dataloaders.aDataset(df_val, input_features=model_params['input_features'],scaler_X = tDs.scaler_X)
testDs = dataloaders.aDataset(df_test, input_features=model_params['input_features'],scaler_X = tDs.scaler_X, Perp=False, assemble_labels=False)

model.eval()
for inputs,labels_train in DataLoader(tDs , shuffle=False, batch_size=testDs.__len__()):
    y_pred_train, gn = model(*inputs)

for inputs,labels_val in DataLoader(vDs , shuffle=False, batch_size=testDs.__len__()):
    y_pred_val, gn = model(*inputs)


#for inputs,labels_test in DataLoader(testDs, shuffle=False, batch_size=testDs.__len__()):
y_pred_test, gn = model(testDs.X,testDs.T)
nut_L = df_test['komegasst_nut']*-gn[:,0].detach().numpy()
gn2 = gn[:,1:]
T = testDs.T[:,1:,:]
b_perp_pred = torch.sum(gn2.view(-1,9,1,1)*torch.ones_like(T)*T,axis=1).detach().numpy()
a_perp_pred = b_perp_pred*2*df_test['DNS_k'].to_numpy()[:,None,None]# Should ideally be DNS_k, but not available as a volume 
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
#print(f'Test losses: {losses.aLoss(y_pred_test, *labels_test)}')

df[f'pred_b_11_all'] = y_pred_train.detach().numpy()[:,0,0]
df[f'pred_b_12_all'] = y_pred_train.detach().numpy()[:,0,1]
df[f'pred_b_13_all'] = y_pred_train.detach().numpy()[:,0,2]
df[f'pred_b_22_all'] = y_pred_train.detach().numpy()[:,1,1]
df[f'pred_b_23_all'] = y_pred_train.detach().numpy()[:,1,2]
df[f'pred_b_33_all'] = y_pred_train.detach().numpy()[:,2,2]

df[f'pred_a_11_all'] = y_pred_train.detach().numpy()[:,0,0]*2*df[f'DNS_k']
df[f'pred_a_12_all'] = y_pred_train.detach().numpy()[:,0,1]*2*df[f'DNS_k']
df[f'pred_a_13_all'] = y_pred_train.detach().numpy()[:,0,2]*2*df[f'DNS_k']
df[f'pred_a_22_all'] = y_pred_train.detach().numpy()[:,1,1]*2*df[f'DNS_k']
df[f'pred_a_23_all'] = y_pred_train.detach().numpy()[:,1,2]*2*df[f'DNS_k']
df[f'pred_a_33_all'] = y_pred_train.detach().numpy()[:,2,2]*2*df[f'DNS_k']



fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,5))
for i, scalar in enumerate(['b_11','b_12','b_13','b_22','b_23','b_33']):
    vmin = min(df[f'DNS_{scalar}'])
    vmax = max(df[f'DNS_{scalar}'])
    axs[0,i].tricontourf(df['komegasst_C_2'],df['komegasst_C_3'],df[f'DNS_{scalar}'])
    axs[1,i].tricontourf(df['komegasst_C_2'],df['komegasst_C_3'],df[f'pred_{scalar}_all'])
    axs[0,i].set_aspect(1)
    axs[1,i].set_aspect(1)
fig.savefig(f'models/duct_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}_b_test.png',dpi=300)

fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,5))
for i, scalar in enumerate(['a_11','a_12','a_13','a_22','a_23','a_33']):
    vmin = min(df[f'DNS_{scalar}'])
    vmax = max(df[f'DNS_{scalar}'])
    axs[0,i].tricontourf(df['komegasst_C_2'],df['komegasst_C_3'],df[f'DNS_{scalar}'])
    axs[1,i].tricontourf(df['komegasst_C_2'],df['komegasst_C_3'],df[f'pred_{scalar}_all'])
    axs[0,i].set_aspect(1)
    axs[1,i].set_aspect(1)


fig.savefig(f'models/duct_only/model_{model.__class__.__name__}_{model_params["n_hidden"]}x{model_params["neurons"]}_a_test.png',dpi=300)


import sys
sys.path.insert(1, '/home/ryley/WDK/ML/code/data_scripting/')
import dataFoam

import numpy as np
from dataFoam.utilities.foamIO.writeFoam_DUCT import writeFoam_ap_DUCT, writeFoam_nut_L_DUCT, writeFoam_genericscalar_DUCT
from dataFoam.utilities.foamIO.readFoam import get_endtime
import os

endtime = '0'
foamdir = '/home/ryley/WDK/ML/scratch/injection/squareDuct_Re_2000_testing_models'
writeFoam_nut_L_DUCT(os.path.join(foamdir,endtime,'nut_L'),nut_L)
writeFoam_ap_DUCT(os.path.join(foamdir,endtime,'aperp'),a_perp_pred)

torch.save(model.state_dict(), f'models/duct_only/duct_only.state_dict')

with open(f"models/params_cluster_duct_only.pickle", 'wb') as handle:
    pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


endtime = '0'
foamdir = '/home/ryley/WDK/ML/scratch/injection/squareDuct_Re_2000_label_injection'
writeFoam_nut_L_DUCT(os.path.join(foamdir,endtime,'nut_L'),df_test['DNS_nut_opt'])
a_perp_label = np.empty((len(df_test['komegasst_nut']),3,3))
a_perp_label[:,0,0] = df_test['DNS_aperp_11']
a_perp_label[:,0,1] = df_test['DNS_aperp_12']
a_perp_label[:,0,2] = df_test['DNS_aperp_13']
a_perp_label[:,1,1] = df_test['DNS_aperp_22']
a_perp_label[:,1,2] = df_test['DNS_aperp_23']
a_perp_label[:,2,1] = df_test['DNS_aperp_33']

a_perp_label[:,1,0] = a_perp_label[:,0,1]
a_perp_label[:,2,0] = a_perp_label[:,0,2]
a_perp_label[:,2,1] = a_perp_label[:,1,2]

writeFoam_ap_DUCT(os.path.join(foamdir,endtime,'aperp'),a_perp_label)
writeFoam_genericscalar_DUCT(os.path.join(foamdir,endtime,'DNS_k'),'DNS_k',df_test['DNS_k'])

    