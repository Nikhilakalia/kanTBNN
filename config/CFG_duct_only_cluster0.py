import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'duct_only_cluster0'
#############################

results_dir = os.path.join('models',run_name)
evaluation = None

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
                  'val_set': ['squareDuctAve_Re_1300','squareDuctAve_Re_1800','squareDuctAve_Re_3200'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['squareDuctAve_Re_2000'],
                  'cluster': 0
                }
#dataset_params['Cases'] = ['case_1p0']

training_params = { 'loss_fn': partial(losses.aLoss),
                    'max_epochs': 500,
                    'learning_rate': 0.0001,
                    'learning_rate_decay': 1.0,
                    'batch_size': 32,
                    'early_stopping_patience': 100,
                    'early_stopping_min_delta': 1E-8,
                }

model_params = {
    'model_type': tbnn.models.TBNNiii,
    'neurons': 30, 
    'n_hidden': 8, 'activation_function': nn.SiLU(),                 
    'input_features':[
'komegasst_q5',
'komegasst_q6',
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
    ]
}

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]