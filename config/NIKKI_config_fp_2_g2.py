import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'kan_6_g2'
#############################

results_dir = os.path.join('models',run_name)
evaluation = tbnn.evaluate.flatplate

dataset_params = {'file': '/home/nikki/kan/data/turbulence_dataset_clean.csv',
                  'Cases': ['fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'],
                  'val_set': ['fp_3030','fp_1410','fp_4060'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['fp_3630']
                }
#dataset_params['Cases'] = ['case_1p0']

training_params = { 'loss_fn': partial(losses.aLoss),
                    'max_epochs': 6000,
                    'learning_rate': 0.002,
                    'learning_rate_decay': 1.0,
                    'batch_size': 64,
                    'early_stopping_patience': 100000,
                    'early_stopping_min_delta': 1E-8,
                }

model_params = {
    'model_type': tbnn.models.kanTBNN,
    'width': [6,6,10], 
    'grid': 6,
     'k': 3,
    'input_features':[
'komegasst_I1_1',
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

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]