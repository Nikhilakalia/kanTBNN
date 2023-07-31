import pandas as pd
import torch.nn as nn
from tbnn.training_utils import early_stopped_tbnn_training_run_k
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('cluster_number')
args = parser.parse_args()
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import tbnn.models as models
import tbnn.devices as devices
device = devices.get_device()

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/komegasst_split.csv',
                  'test_set': ['fp_3630'],
                }

df = pd.read_csv(dataset_params['file'])
args = parser.parse_args()

df = df[df['Case'] == 'fp_3630']
df_test = df
df_tv = df
print(f'Dataset: {len(df)}, test: {len(df_test)}, tv: {len(df_tv)}')

training_params = {'early_stopping_patience': 20, 'max_epochs': 1000, 'learning_rate': 0.0001, 'learning_rate_decay': 0.99, 'batch_size': 128, 'val_set': ['case_1p5', 'fp_1410', 'fp_3030']}

training_params['val_set']=['fp_3630']
model_params = {'neurons': 20, 'n_hidden': 2, 'activation_function': nn.SiLU(), 'input_features': ['komegasst_I1_1', 'komegasst_I1_3', 'komegasst_I2_3', 'komegasst_I1_5']}

model = models.TBNN_k(N = 10,
                input_dim = len(model_params['input_features']),
                n_hidden = model_params['n_hidden'],
                neurons = model_params['neurons'],
                activation_function = model_params['activation_function'],
                input_feature_names=model_params['input_features']
            ).to(device)

#model, loss_vals, val_loss_vals  = early_stopped_tbnn_training_run(model_params,training_params,df_tv)
model, loss_vals, val_loss_vals  = early_stopped_tbnn_training_run_k(model = model,
                                                                   training_params = training_params,
                                                                   df_tv = df_tv)

#torch.save(model.state_dict(), f'models/cluster_{cluster}')


#with open(f"models/params_cluster_{cluster}.pickle", 'wb') as handle:
#    pickle.dump(model_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig, ax = plt.subplots(1,figsize=(5,5))
ax.plot(loss_vals,'-',color='b')
ax.plot(val_loss_vals,'--',color='r')
ax.semilogy()
fig.tight_layout()
#fig.savefig(f'models/model_cluster_{cluster}.png',dpi=300)