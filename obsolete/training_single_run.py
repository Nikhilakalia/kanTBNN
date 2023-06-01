import pandas as pd
import torch.nn as nn
from training_utils import early_stopped_tbnn_training_run

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/komegasst_split.csv',
                  'test_set': ['case_1p2','fp_3630'],
                  'final_val_set': ['case_1p0','fp_2000','fp_3970'],
                  'cv_val_sets': {0: ['case_0p5','fp_2540','fp_1000'],
                                  1: ['case_0p8','fp_4060','fp_3270'],
                                  2: ['case_1p5','fp_1410','fp_3030']}
                }

df = pd.read_csv(dataset_params['file'])
df_test = df[df['Case'].isin(dataset_params['test_set'])]
df_final_val = df[df['Case'].isin(dataset_params['final_val_set'])]
df_tv = df[~df['Case'].isin(dataset_params['test_set']+dataset_params['final_val_set'])]
print(f'Dataset: {len(df)}, test: {len(df_test)}, final_val: {len(df_final_val)}, tv: {len(df_tv)}')

training_params = {'early_stopping_patience': 20,
                'max_epochs': 1000,
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
results = early_stopped_tbnn_training_run(model_params,training_params,df_tv)