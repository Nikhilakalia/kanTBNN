import pandas as pd
import torch.nn as nn
from training_utils import early_stopped_tbnn_training_run
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('cluster_number')
args = parser.parse_args()
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
dataset_params = {'file': '/home/ryley/WDK/ML/dataset/komegasst_split.csv',
                  'test_set': ['case_1p2','fp_3630'],
                }

df = pd.read_csv(dataset_params['file'])
args = parser.parse_args()
cluster = args.cluster_number

if cluster != 'all':
    cluster = int(args.cluster_number)
    df = df[df['Cluster']==cluster]
else:
    cluster = args.cluster_number
df_test = df[df['Case'].isin(dataset_params['test_set'])]
df_tv = df[~df['Case'].isin(dataset_params['test_set'])]
print(f'Dataset: {len(df)}, test: {len(df_test)}, tv: {len(df_tv)}')

training_params = {'early_stopping_patience': 20, 'max_epochs': 1000, 'learning_rate': 0.0001, 'learning_rate_decay': 0.99, 'batch_size': 128, 'val_set': ['case_1p5', 'fp_1410', 'fp_3030']}

training_params['val_set']=['case_1p0','fp_2000','fp_3970']
model_params = {'neurons': 50, 'n_hidden': 3, 'activation_function': nn.SiLU(), 'input_features': ['komegasst_I1_1', 'komegasst_I1_3', 'komegasst_I2_3', 'komegasst_I1_5']}


model, loss_vals, val_loss_vals  = early_stopped_tbnn_training_run(model_params,training_params,df_tv)
torch.save(model.state_dict(), f'models/cluster_{cluster}')


with open(f"models/params_cluster_{cluster}.pickle", 'wb') as handle:
    pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig, ax = plt.subplots(1,figsize=(5,5))
ax.plot(loss_vals,'-',color='b')
ax.plot(val_loss_vals,'--',color='r')
ax.semilogy()
fig.tight_layout()
fig.savefig(f'models/model_cluster_{cluster}.png',dpi=300)