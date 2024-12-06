import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

run_name = 'kan_experiment_13'
results_dir = '/home/nikki/kan/kanTBNN/models/multi_run/kan_experiment_13'
if not os.path.exists(results_dir): os.makedirs(results_dir)
evaluation = tbnn.evaluate.flatplate

dataset_params = {'file': '/home/nikki/kan/data/turbulence_dataset_clean.csv', 'Cases': ['fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'], 'val_set': ['fp_3030', 'fp_1410', 'fp_4060'], 'test_set': ['fp_3630']}
training_params = {
    'loss_fn': partial(losses.aLoss),
    'max_epochs': 6000,
    'learning_rate': 0.001,
    'learning_rate_decay': 1.0,
    'batch_size': 64,
    'early_stopping_patience': 100000,
    'early_stopping_min_delta': 1e-08,
}

model_params = {
    'model_type': tbnn.models.kanTBNN,
    'width': [6, 7, 10],
    'grid': 4,
    'k': 3,
    'input_features': ['komegasst_I1_16', 'komegasst_I1_10', 'komegasst_I1_9', 'komegasst_I1_5', 'komegasst_I1_7', 'komegasst_I1_3'],
}

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]
