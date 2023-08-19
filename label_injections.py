import pandas as pd
import torch.nn as nn
from tbnn.training_utils import early_stopped_tbnn_training_run
import torch
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import tbnn.models as models
import tbnn.devices as devices
import tbnn.dataloaders as dataloaders
import sys
import tbnn.losses as losses
import numpy as np

import sys
sys.path.insert(1, '/home/ryley/WDK/ML/code/data_upgrader/')
import dataFoam

from dataFoam.utilities.foamIO.writeFoam_DUCT import writeFoam_ap_DUCT, writeFoam_nut_L_DUCT, writeFoam_genericscalar_DUCT
from dataFoam.utilities.foamIO.writeFoam_PHLL import writeFoam_ap_PHLL, writeFoam_nut_L_PHLL, writeFoam_genericscalar_PHLL

from dataFoam.utilities.foamIO.readFoam import get_endtime
import os
device = devices.get_device()

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_raw_squareDuctFull_Re_2000.csv',
                  'Cases': ['squareDuctFull_Re_2000']
                }

df = pd.read_csv(dataset_params['file'])
df = df[df.Case.isin(dataset_params['Cases'])]

endtime = '0'
foamdir = '/home/ryley/WDK/ML/scratch/injection/squareDuct_Re_2000_label_injection'
writeFoam_nut_L_DUCT(os.path.join(foamdir,endtime,'nut_L'),df['komegasst_nut'])
a_perp_label = np.empty((len(df['komegasst_nut']),3,3))
a_perp_label[:,0,0] = df['DNS_aperp_11']
a_perp_label[:,0,1] = df['DNS_aperp_12']
a_perp_label[:,0,2] = df['DNS_aperp_13']
a_perp_label[:,1,1] = df['DNS_aperp_22']
a_perp_label[:,1,2] = df['DNS_aperp_23']
a_perp_label[:,2,1] = df['DNS_aperp_33']

a_perp_label[:,1,0] = a_perp_label[:,0,1]
a_perp_label[:,2,0] = a_perp_label[:,0,2]
a_perp_label[:,2,1] = a_perp_label[:,1,2]

writeFoam_ap_DUCT(os.path.join(foamdir,endtime,'aperp'),a_perp_label)
writeFoam_genericscalar_DUCT(os.path.join(foamdir,endtime,'DNS_k'),'DNS_k',df['DNS_k'])


###################### PERIODIC HILL

dataset_params = {'file': '/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv',
                  'Cases': ['case_1p2']
                }

df = pd.read_csv(dataset_params['file'])
df = df[df.Case.isin(dataset_params['Cases'])]

endtime = '0'
foamdir = '/home/ryley/WDK/ML/scratch/injection/case_1p2_label_injection'
writeFoam_nut_L_PHLL(os.path.join(foamdir,endtime,'nut_L'),df['komegasst_nut'])
a_perp_label = np.empty((len(df['komegasst_nut']),3,3))
a_perp_label[:,0,0] = df['DNS_aperp_11']
a_perp_label[:,0,1] = df['DNS_aperp_12']
a_perp_label[:,0,2] = df['DNS_aperp_13']
a_perp_label[:,1,1] = df['DNS_aperp_22']
a_perp_label[:,1,2] = df['DNS_aperp_23']
a_perp_label[:,2,1] = df['DNS_aperp_33']

a_perp_label[:,1,0] = a_perp_label[:,0,1]
a_perp_label[:,2,0] = a_perp_label[:,0,2]
a_perp_label[:,2,1] = a_perp_label[:,1,2]

writeFoam_ap_PHLL(os.path.join(foamdir,endtime,'aperp'),a_perp_label)
#writeFoam_genericscalar_PHLL(os.path.join(foamdir,endtime,'DNS_k'),'DNS_k',df['DNS_k'])