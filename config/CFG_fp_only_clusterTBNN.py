import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'fp_only_clusterTBNN'
#############################

results_dir = os.path.join('models',run_name)
evaluation = tbnn.evaluate.flatplate

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean_split.csv',
                  'Cases': ['fp_1000', 'fp_1410', 'fp_2000', 'fp_2540', 'fp_3030', 'fp_3270', 'fp_3630', 'fp_3970', 'fp_4060'],
                  'val_set': ['fp_3030','fp_1410','fp_4060'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['fp_3630'],
                }

training_params = { 'loss_fn': partial(losses.aLoss),
                }

model_params = {
    'model_type': tbnn.models.clusterTBNN,
    'splitr': 'models/splitr/splitr.pkl',
    'splitr_scaler': 'models/splitr/splitr_scaler.pkl',
    'splitr_input_features': ['komegasst_I1_1','komegasst_I1_3','komegasst_I1_5','komegasst_q5','komegasst_q6'],
    'models': {
        'model_cluster0': 'models/fp_only_cluster0/TBNNiii-23Aug19_15:41:57.pickle',
        'model_cluster1': 'models/fp_only_cluster1/TBNNiii-23Aug19_16:07:38.pickle'
        }
}
dataset_params['data_loader'] = tbnn.dataloaders.aDataset
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]