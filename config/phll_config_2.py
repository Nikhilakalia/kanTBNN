import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

run_name = 'phll_experiment_2'
results_dir = '/home/nikki/kan/kanTBNN/models/multi_run_phll/phll_experiment_2'
evaluation = tbnn.evaluate.periodic_hills

dataset_params = {'file': '/home/nikki/kan/data/turbulence_dataset_clean.csv', 'Cases': ['case_0p5', 'case_0p8', 'case_1p0', 'case_1p2', 'case_1p5'], 'val_set': ['case_0p8'], 'test_set': ['case_1p2']}

training_params = {
    'loss_fn': partial(losses.aLoss, alpha=100),
    'max_epochs': 20000,
    'learning_rate': 8.06180284596018e-06,
    'learning_rate_decay': 1.0,
    'batch_size': 128,
    'early_stopping_patience': 5000,
    'early_stopping_min_delta': 1e-08,
}

model_params = {
    'model_type': tbnn.models.kanTBNN,
    'width': [17, 13, 15, 10, 10],
    'grid': 10,
    'k': 3,
    'input_features': ['komegasst_q6', 'komegasst_q5', 'komegasst_q8', 'komegasst_I1_16', 'komegasst_I1_7', 'komegasst_I1_3', 'komegasst_I2_6', 'komegasst_q3', 'komegasst_I2_3', 'komegasst_I1_4', 'komegasst_I2_7', 'komegasst_I1_35', 'komegasst_q2', 'komegasst_q7', 'komegasst_I1_1', 'komegasst_q4', 'komegasst_I2_8'],
}

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]
