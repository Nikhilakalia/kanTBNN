import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'duct_only_3'
#############################

results_dir = os.path.join('models',run_name)
evaluation = tbnn.evaluate.square_duct

dataset_params = {'file': '/home/nikki/kan/data/turbulence_dataset_clean.csv',
                  'Cases': ['squareDuctAve_Re_1100',
       'squareDuctAve_Re_1150',
       #'squareDuctAve_Re_1250',
       'squareDuctAve_Re_1300',
       #'squareDuctAve_Re_1350',
       'squareDuctAve_Re_1400',
       #'squareDuctAve_Re_1500',
       'squareDuctAve_Re_1600',
       'squareDuctAve_Re_1800',
       'squareDuctAve_Re_2000',
       #'squareDuctAve_Re_2205',
       'squareDuctAve_Re_2400',
       'squareDuctAve_Re_2600',
       #'squareDuctAve_Re_2900',
       'squareDuctAve_Re_3200',
       #'squareDuctAve_Re_3500'
        ],
                  'val_set': ['squareDuctAve_Re_1300','squareDuctAve_Re_1800','squareDuctAve_Re_3200'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['squareDuctAve_Re_2000'],
                }
#dataset_params['Cases'] = ['case_1p0']

training_params = { 'loss_fn': partial(losses.aLoss,alpha=100),
                    'max_epochs': 500,
                    'learning_rate': 0.0005,
                    'learning_rate_decay': 1.0,
                    'batch_size': 32,
                    'early_stopping_patience': 500,
                    'early_stopping_min_delta': 1E-8,
                }

model_params = {
    'model_type': tbnn.models.kanTBNN,
    'width': [6,3,10], #change the width
    'grid': 3, #can also change the grid for the thickness of the network 
     'k': 3,
    'input_features':[
#'komegasst_I1_1',
#'komegasst_I1_3',
#'komegasst_I1_4',
'komegasst_I1_5',
#'komegasst_I1_16',
'komegasst_I1_7',
#'komegasst_I1_9',
'komegasst_I1_10',
'komegasst_I1_12',
#'komegasst_I1_13',
'komegasst_I1_16',
'komegasst_q5',
#'komegasst_q6',
    ]
}


dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]