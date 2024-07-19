import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'phll_memorization'
#############################

results_dir = os.path.join('models',run_name)
evaluation = tbnn.evaluate.periodic_hills_k

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv',
                  'Cases': ['case_1p2'],
                  'val_set': ['case_1p2'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['case_1p2']
                }
#dataset_params['Cases'] = ['case_1p0']

training_params = { 'loss_fn': partial(losses.mse_Delta),
                    'max_epochs': 3000,
                    'learning_rate': 0.0001,
                    'learning_rate_decay': 1.0,
                    'batch_size': 32,
                    'early_stopping_patience': 500,
                    'early_stopping_min_delta': 1E-8,
                }

model_params = {
    'model_type': tbnn.models.KCNN,
    'neurons': 20, 
    'n_hidden': 5, 'activation_function': nn.SiLU(),                 
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
    ]
}

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]