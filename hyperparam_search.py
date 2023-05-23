import pandas as pd
import torch.nn as nn
from training_utils import early_stopped_tbnn_training_run
import json
import matplotlib.pyplot as plt
import argparse

def update_params(original_params, new_params):
    updated_params = original_params
    for param in original_params.keys():
        if param in new_params.keys():
            updated_params[param] = new_params[param]
        else:
            pass
    return updated_params 

with open('param_list.json') as param_list_file:
    param_list = json.load(param_list_file)

parser = argparse.ArgumentParser()
parser.add_argument('cluster_number')

dataset_params = {'file': 'komegasst_split.csv',
                  'test_set': ['case_1p2','fp_3630'],
                  'final_val_set': ['case_1p0','fp_2000','fp_3970'],
                  'cv_val_sets': {0: ['case_0p5','fp_2540','fp_1000'],
                                  1: ['case_0p8','fp_4060','fp_3270'],
                                  2: ['case_1p5','fp_1410','fp_3030']}
                }

df = pd.read_csv(dataset_params['file'])

args = parser.parse_args()
cluster = args.cluster_number

if cluster != 'all':
    cluster = int(args.cluster_number)
    df = df[df['Cluster']==cluster]
else:
    cluster = args.cluster_number

print(f'CLUSTER: {cluster}')

df_test = df[df['Case'].isin(dataset_params['test_set'])]
df_final_val = df[df['Case'].isin(dataset_params['final_val_set'])]
df_tv = df[~df['Case'].isin(dataset_params['test_set']+dataset_params['final_val_set'])]
print(f'Dataset total: {len(df)}, test: {len(df_test)}, final_val: {len(df_final_val)}, tv: {len(df_tv)}')

training_params = {'early_stopping_patience': 20,
                'max_epochs': 20,
                'learning_rate': 0.001,
                'learning_rate_decay': 0.98,
                'batch_size': 128,
                'val_set':['case_1p5','fp_1410','fp_3030'],
               }

model_params = {'neurons': 50,
                'n_hidden': 5,
                'activation_function': nn.SiLU(),
                'input_features': ['komegasst_q6','komegasst_q5','komegasst_I1_1','komegasst_I1_3','komegasst_I2_3','komegasst_I1_5']
}


best_cv_score = 1E10
for search_iter, params in enumerate(param_list):
    print('\n====================================================')
    print(f'                SEARCH ITERATION: {search_iter} ')
    print('====================================================\n')

    training_params = update_params(training_params, params)
    model_params = update_params(model_params, params)
    loss_values = dict.fromkeys(dataset_params['cv_val_sets'].keys())
    val_loss_values = dict.fromkeys(dataset_params['cv_val_sets'].keys())

    cv_score = 0.0
    for i, cv_val_set_i in enumerate(dataset_params['cv_val_sets'].keys()):
        print(f'\n=====| CV SET: {i}\n')
        training_params['val_set'] = dataset_params['cv_val_sets'][cv_val_set_i]
        model, loss_values[i], val_loss_values[i] = early_stopped_tbnn_training_run(model_params,training_params,df_tv)
        cv_score += val_loss_values[i][-1]
    
    cv_score /= len(dataset_params['cv_val_sets'].keys())

    if cv_score < best_cv_score:
        best_cv_score = cv_score
        print(f'\n=============> NEW BEST CV SCORE FOUND: {cv_score} <==============\nPARAMS:')
        print(training_params)
        print(model_params)
        best_loss_values = loss_values
        best_val_loss_values = val_loss_values
        fig, ax = plt.subplots(1,figsize=(5,5))
        colors = ['b','r','g']
        for set_i in dataset_params['cv_val_sets'].keys():
            ax.plot(best_loss_values[set_i][-100:],'-',color=colors[set_i])
            ax.plot(best_val_loss_values[set_i][-100:],'--',color=colors[set_i])
        ax.semilogy()
        fig.tight_layout()
        fig.savefig(f'best_model_cluster_{cluster}.png',dpi=300)
    #results = early_stopped_tbnn_training_run(model_params,training_params,df_tv)