import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

run_name = 'kan_experiment_17_opt'
results_dir = '/home/nikhila/WDC/kan/kanTBNN/models/multi_run/kan_experiment_17_opt'
evaluation = tbnn.evaluate.square_duct

dataset_params = {'file': '/home/nikhila/WDC/kan/dataset/turbulence_dataset_clean.csv', 'Cases': ['squareDuctAve_Re_1100', 'squareDuctAve_Re_1150', 'squareDuctAve_Re_1250', 'squareDuctAve_Re_1300', 'squareDuctAve_Re_1350', 'squareDuctAve_Re_1400', 'squareDuctAve_Re_1500', 'squareDuctAve_Re_1600', 'squareDuctAve_Re_1800', 'squareDuctAve_Re_2000', 'squareDuctAve_Re_2205', 'squareDuctAve_Re_2400', 'squareDuctAve_Re_2600', 'squareDuctAve_Re_2900', 'squareDuctAve_Re_3200', 'squareDuctAve_Re_3500'], 'val_set': ['squareDuctAve_Re_1300', 'squareDuctAve_Re_1800', 'squareDuctAve_Re_3200'], 'test_set': ['squareDuctAve_Re_2000']}
training_params = {
    'loss_fn': partial(losses.aLoss, alpha=100),
    'max_epochs': 500,
    'learning_rate': 0.00014520497367333474,
    'batch_size': 64,
    'early_stopping_patience': 500,
    'early_stopping_min_delta': 1e-08,
    'learning_rate_decay': 1.0,
}

model_params = {
    'model_type': tbnn.models.kanTBNN,
    'width': [16, 9, 5, 5, 10],
    'grid': 7,
    'k': 3,
    'input_features': ['komegasst_I1_1', 'komegasst_I1_3', 'komegasst_I1_4', 'komegasst_I1_5', 'komegasst_I1_7', 'komegasst_I1_9', 'komegasst_I1_10', 'komegasst_I1_12', 'komegasst_I1_13', 'komegasst_I1_16', 'komegasst_q5', 'komegasst_q6', 'komegasst_q2', 'komegasst_q3', 'komegasst_q4', 'komegasst_I2_6'],
}

dataset_params['data_loader'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[0]
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]
