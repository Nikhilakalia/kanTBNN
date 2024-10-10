import torch.nn as nn
import tbnn.losses as losses
import tbnn
from functools import partial
import os

#############################
run_name = 'phll_only'
#############################
barcode_TBNN = 'TBNNiii-24May30_22:10:42' #'TBNNiii-24May29_23:17:59'
barcode_KCNN = 'KCNN-24May28_16:28:33'

results_dir = os.path.join('models',run_name)
evaluation = tbnn.evaluate.periodic_hills_injection
