import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'all_flows'
#############################

results_dir = os.path.join('models',run_name)
evaluation = [tbnn.evaluate.flatplate_k,
              tbnn.evaluate.square_duct_k,
              tbnn.evaluate.periodic_hills_k]

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv',
                  'Cases': ['case_0p5','case_0p8','case_1p0','case_1p2','case_1p5',
                  'squareDuctAve_Re_1100',
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
                  'squareDuctAve_Re_3500',
                  'fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'                  
                  ],
                  'val_set': ['fp_3030','fp_1410','fp_4060','squareDuctAve_Re_1300','squareDuctAve_Re_1800','squareDuctAve_Re_3200','case_0p8'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['fp_3630','squareDuctAve_Re_2000','case_1p2'],
                }
#dataset_params['Cases'] = ['case_1p0']

training_params = { 'loss_fn': partial(losses.mse_Delta),
                    'max_epochs': 1000, # was 5000
                    'learning_rate': 0.0005, # was 0.00005
                    'learning_rate_decay': 1.0,
                    'batch_size': 32,# was 128?
                    'early_stopping_patience': 10000,
                    'early_stopping_min_delta': 1E-8,
                }

model_params = {
    'model_type': tbnn.models.KCNN,
    'neurons': 40, 
    'n_hidden': 10, 'activation_function': nn.SiLU(),                 
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
  #'komegasst_q7',
  'komegasst_I1_1',
  'komegasst_q4',
  'komegasst_I2_8',
    ]
}

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]