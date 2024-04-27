import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'phll_only_clusterTBNN'
#############################

results_dir = os.path.join('models',run_name)
evaluation = tbnn.evaluate.periodic_hills

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean_split.csv',
                  'Cases': ['case_0p5','case_0p8','case_1p0','case_1p2','case_1p5'],
                  'val_set': ['case_0p8'],#['squareDuctQuad1_Re_1300','squareDuctQuad1_Re_1800','squareDuctQuad1_Re_3200'],
                  'test_set': ['case_1p2'],
                }

training_params = { 'loss_fn': partial(losses.aLoss),
                }

model_params = {
    'model_type': tbnn.models.clusterTBNN,
    'splitr': 'models/splitr/splitr.pkl',
    'splitr_scaler': 'models/splitr/splitr_scaler.pkl',
    'splitr_input_features': ['komegasst_I1_1','komegasst_I1_3','komegasst_I1_5','komegasst_q5','komegasst_q6'],
    'models': {
        'model_cluster0': 'models/phll_only_cluster0/TBNNiii-23Aug20_01:20:55.pickle',
        'model_cluster1': 'models/phll_only_cluster1/TBNNiii-23Aug20_01:23:52.pickle',
        'model_cluster2': 'models/phll_only_cluster2/TBNNiii-23Aug20_01:31:02.pickle',
        'model_cluster3': 'models/phll_only_cluster3/TBNNiii-23Aug20_01:52:47.pickle',
        }
}
dataset_params['data_loader'] = tbnn.dataloaders.aDataset
training_params['mseLoss'] = tbnn.training_utils.get_dataloader_type(training_params['loss_fn'])[1]