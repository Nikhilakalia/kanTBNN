import tbnn.losses as losses
import tbnn.training_utils as training_utils
from torch.utils.data import DataLoader
import tbnn.dataloaders as dataloaders
import torch
import matplotlib.pyplot as plt
import pandas as pd
import tbnn.predict as predict
import dataFoam

import numpy as np
from dataFoam.utilities.foamIO.writeFoam_PHLL import writeFoam_ap_PHLL, writeFoam_nut_L_PHLL, writeFoam_genericscalar_PHLL
from dataFoam.utilities.foamIO.writeFoam_DUCT import writeFoam_ap_DUCT, writeFoam_nut_L_DUCT, writeFoam_genericscalar_DUCT
from dataFoam.utilities.foamIO.readFoam import get_endtime
import os

def final_model(model, dataframes, config):
    """
    Summarize model performance at the end of training.
    """
    loss_fn = config.training_params['loss_fn']
    print('Final model performance:')
    names = ['Training','Validation','Testing']
    assert len(dataframes) <= 3 

    for i, dfi in enumerate(dataframes):
        g_pred, b_pred, b_perp_pred, a_pred, a_perp_pred = predict.TBNN(model,dfi)
        ds = config.dataset_params['data_loader'](dfi, input_features=None)
        for _,labels in DataLoader(ds , shuffle=False, batch_size=ds.__len__()):
            print(f'======= {names[i]} =======')
            print(f'{config.training_params["loss_fn"].func.__name__}:   {loss_fn(b_pred, g_pred, *labels):.4E}')
            print(f'mse_b:   {losses.mseLoss(b_pred, labels[-2]).item():.4E}')
            print(f'realzLoss: {losses.realizabilityLoss(b_pred).item():.4E}')
            print(f'%unrealiz: {training_utils.count_nonrealizable(b_pred)/len(b_pred)*100:.4f}%')
        ds = dataloaders.aDataset(dfi, input_features=None)
        for _,labels in DataLoader(ds , shuffle=False, batch_size=ds.__len__()):
            print(f'mse_a (khat):   {losses.mseLoss_aLoss(b_pred, g_pred, *labels).item():.4E}')

def intermediate_model(model, datasets, loss_fn):
    """
    Return intermediate train/valid loss values during training.
    """
    for inputs,labels in DataLoader(datasets[0] , shuffle=False, batch_size=datasets[0].__len__()):
        y_pred, g_pred = model(*inputs)
        loss_t = loss_fn(y_pred, g_pred, *labels).item()

    for inputs,labels in DataLoader(datasets[1] , shuffle=False, batch_size=datasets[1].__len__()):
        y_pred, g_pred = model(*inputs)
        loss_v = loss_fn(y_pred, g_pred, *labels).item()

    return loss_t, loss_v
 
def print_intermediate_info(model, datasets, loss_fn, mseLoss, epoch, lr):
    """
    Add a line to the learning loss table.
    """
    loss_t, loss_v = intermediate_model(model, datasets, loss_fn)
    for inputs,labels in DataLoader(datasets[0] , shuffle=False, batch_size=datasets[0].__len__()):
        y_pred_t, g_pred = model(*inputs)
        mse_t = mseLoss(y_pred_t, g_pred, *labels).item()
        rl_t = losses.realizabilityLoss(y_pred_t).item()  \
        
    for inputs,labels in DataLoader(datasets[1] , shuffle=False, batch_size=datasets[1].__len__()):
        y_pred_v, g_pred = model(*inputs)
        mse_v = mseLoss(y_pred_v, g_pred, *labels).item()
        rl_v = losses.realizabilityLoss(y_pred_v).item()  
        print(f"{epoch:3d}   "
                f"{lr:.3E}   "
                f"{loss_t:.4E}   "
                f"{loss_v:.4E}   "
                f"{mse_t:.4E} / {mse_v:.4E}   "
                f"{rl_t:.4E} / {rl_v:.4E}   "
                f"{training_utils.count_nonrealizable(y_pred_t)/len(y_pred_t)*100:.2f}% / {training_utils.count_nonrealizable(y_pred_v)/len(y_pred_v)*100:.2f}%",
                flush=True)
  
def periodic_hills(model, config):
    print('===================================')
    print('=====PERIODIC HILLS EVALUATION=====')
    print('===================================')
    data_loader = config.dataset_params['data_loader']
    df, df_train, df_valid, df_test = training_utils.get_dataframes(config.dataset_params,print_info=True)

    df = pd.read_csv('/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv')
    df_test = df[df['Case'] == 'case_1p2'].copy()
    final_model(model,[df_train, df_valid, df_test],config)
    gn, b_pred, b_perp_pred, a_pred, a_perp_pred = predict.TBNN(model,df_test,k_name='komegasst_k')

    nut_L = df_test['komegasst_nut'].to_numpy()

    df_test = add_predictions_to_df(gn.detach().numpy(),
                                    b_pred.detach().numpy(),
                                    b_perp_pred.detach().numpy(),
                                    a_pred, a_perp_pred,
                                    df_test)
    fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,3))
    for i, scalar in enumerate(['b_11','b_12','b_13','b_22','b_23','b_33']):
        vmin = min(df_test[f'DNS_{scalar}'])
        vmax = max(df_test[f'DNS_{scalar}'])
        axs[0,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'DNS_{scalar}'],vmin = vmin, vmax = vmax)
        axs[1,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'pred_{scalar}'],vmin = vmin, vmax = vmax)
        axs[0,i].set_aspect(1)
        axs[1,i].set_aspect(1)
    fig.savefig(os.path.join(config.results_dir,f'{model.barcode}_phll_b_test.png'),dpi=300)

    fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,3))
    for i, scalar in enumerate(['a_11','a_12','a_13','a_22','a_23','a_33']):
        vmin = min(df_test[f'DNS_{scalar}'])
        vmax = max(df_test[f'DNS_{scalar}'])
        axs[0,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'DNS_{scalar}'],vmin = vmin, vmax = vmax)
        axs[1,i].tricontourf(df_test['komegasst_C_1'],df_test['komegasst_C_2'],df_test[f'pred_{scalar}'],vmin = vmin, vmax = vmax)
        axs[0,i].set_aspect(1)
        axs[1,i].set_aspect(1)

    fig.savefig(os.path.join(config.results_dir,f'{model.barcode}_phll_a_test.png'),dpi=300)
        
    endtime = '0'
    foamdir = os.path.join(f'/home/ryley/WDK/ML/scratch/injection/case_1p2_{config.run_name}')
    writeFoam_nut_L_PHLL(os.path.join(foamdir,endtime,'nut_L'),nut_L)
    writeFoam_ap_PHLL(os.path.join(foamdir,endtime,'aperp'),a_perp_pred)

def square_duct(model,config):
    print('===================================')
    print('======SQUARE DUCT EVALUATION=======')
    print('===================================')
    data_loader = config.dataset_params['data_loader']
    df, df_train, df_valid, df_test = training_utils.get_dataframes(config.dataset_params,print_info=True)

    #Select only the square duct test case
    df = pd.read_csv('/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv')
    df_test = df[df['Case'] == 'squareDuctAve_Re_2000'].copy()

    final_model(model,[df_train, df_valid, df_test],config)
    gn, b_pred, b_perp_pred, a_pred, a_perp_pred = predict.TBNN(model,df_test, k_name = 'komegasst_k')

    nut_L = df_test['komegasst_nut'].to_numpy()

    df_test = add_predictions_to_df(gn.detach().numpy(),
                                    b_pred.detach().numpy(),
                                    b_perp_pred.detach().numpy(),
                                    a_pred, a_perp_pred,
                                    df_test)
    fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,3))
    for i, scalar in enumerate(['b_11','b_12','b_13','b_22','b_23','b_33']):
        vmin = min(df_test[f'DNS_{scalar}'])
        vmax = max(df_test[f'DNS_{scalar}'])
        axs[0,i].tricontourf(df_test['komegasst_C_2'],df_test['komegasst_C_3'],df_test[f'DNS_{scalar}'],vmin=vmin,vmax=vmax)
        axs[1,i].tricontourf(df_test['komegasst_C_2'],df_test['komegasst_C_3'],df_test[f'pred_{scalar}'],vmin=vmin,vmax=vmax)
        axs[0,i].set_aspect(1)
        axs[1,i].set_aspect(1)
    fig.savefig(os.path.join(config.results_dir,f'{model.barcode}_duct_b_test.png'),dpi=300)

    fig, axs = plt.subplots(nrows=2,ncols=6,figsize=(15,3))
    for i, scalar in enumerate(['a_11','a_12','a_13','a_22','a_23','a_33']):
        vmin = min(df_test[f'DNS_{scalar}'])
        vmax = max(df_test[f'DNS_{scalar}'])
        axs[0,i].tricontourf(df_test['komegasst_C_2'],df_test['komegasst_C_3'],df_test[f'DNS_{scalar}'],vmin=vmin,vmax=vmax)
        axs[1,i].tricontourf(df_test['komegasst_C_2'],df_test['komegasst_C_3'],df_test[f'pred_{scalar}'],vmin=vmin,vmax=vmax)
        axs[0,i].set_aspect(1)
        axs[1,i].set_aspect(1)

    fig.savefig(os.path.join(config.results_dir,f'{model.barcode}_duct_a_test.png'),dpi=300)
    
    df_test = pd.read_csv('/home/ryley/WDK/ML/dataset/turbulence_dataset_raw_squareDuctFull_Re_2000.csv')
    gn, b_pred, b_perp_pred, a_pred, a_perp_pred = predict.TBNN(model,df_test, k_name='komegasst_k')
    nut_L = df_test['komegasst_nut'].to_numpy()

    endtime = '0'
    foamdir = os.path.join(f'/home/ryley/WDK/ML/scratch/injection/squareDuct_Re_2000_{config.run_name}')
    writeFoam_nut_L_DUCT(os.path.join(foamdir,endtime,'nut_L'),nut_L)
    writeFoam_ap_DUCT(os.path.join(foamdir,endtime,'aperp'),a_perp_pred)

def flatplate(model,config):
    print('===================================')
    print('=======FLAT PLATE EVALUATION=======')
    print('===================================')
    data_loader = config.dataset_params['data_loader']
    df, df_train, df_valid, df_test = training_utils.get_dataframes(config.dataset_params,print_info=True)

    #Select only the periodic hills test case
    df = pd.read_csv('/home/ryley/WDK/ML/dataset/turbulence_dataset_clean.csv')
    df_test = df[df['Case'] == 'fp_3630'].copy()

    final_model(model,[df_train, df_valid, df_test],config)
    gn, b_pred, b_perp_pred, a_pred, a_perp_pred = predict.TBNN(model,df_test, k_name='komegasst_k')

    #nut_L = -df_test['komegasst_nut'].to_numpy()

    #a_perp_pred = b_perp_pred*2*(gn[:,-1]*df_test['DNS_k'].to_numpy())[:,None,None]

    df_test = add_predictions_to_df(gn.detach().numpy(),
                                    b_pred.detach().numpy(),
                                    b_perp_pred.detach().numpy(),
                                    a_pred, a_perp_pred,
                                    df_test)

    fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(15,15))
    axs[0,0].scatter(df_test['komegasst_C_2'],df_test['DNS_b_11'])
    axs[0,0].scatter(df_test['komegasst_C_2'],df_test['pred_b_11'])

    axs[0,1].scatter(df_test['komegasst_C_2'],df_test['DNS_b_12'])
    axs[0,1].scatter(df_test['komegasst_C_2'],df_test['pred_b_12'])

    axs[0,2].scatter(df_test['komegasst_C_2'],df_test['DNS_b_13'])
    axs[0,2].scatter(df_test['komegasst_C_2'],df_test['pred_b_13'])

    axs[1,1].scatter(df_test['komegasst_C_2'],df_test['DNS_b_22'])
    axs[1,1].scatter(df_test['komegasst_C_2'],df_test['pred_b_22'])

    axs[1,2].scatter(df_test['komegasst_C_2'],df_test['DNS_b_23'])
    axs[1,2].scatter(df_test['komegasst_C_2'],df_test['pred_b_23'])

    axs[2,2].scatter(df_test['komegasst_C_2'],df_test['DNS_b_33'])
    axs[2,2].scatter(df_test['komegasst_C_2'],df_test['pred_b_33'])

    fig.savefig(os.path.join(config.results_dir,f'{model.barcode}_fp_b_test.png'),dpi=300)

    fig, axs = plt.subplots(nrows=3,ncols=3,figsize=(15,15))
    axs[0,0].scatter(df_test['komegasst_C_2'],df_test['DNS_a_11'])
    axs[0,0].scatter(df_test['komegasst_C_2'],df_test['pred_a_11'])

    axs[0,1].scatter(df_test['komegasst_C_2'],df_test['DNS_a_12'])
    axs[0,1].scatter(df_test['komegasst_C_2'],df_test['pred_a_12'])

    axs[0,2].scatter(df_test['komegasst_C_2'],df_test['DNS_a_13'])
    axs[0,2].scatter(df_test['komegasst_C_2'],df_test['pred_a_13'])

    axs[1,1].scatter(df_test['komegasst_C_2'],df_test['DNS_a_22'])
    axs[1,1].scatter(df_test['komegasst_C_2'],df_test['pred_a_22'])

    axs[1,2].scatter(df_test['komegasst_C_2'],df_test['DNS_a_23'])
    axs[1,2].scatter(df_test['komegasst_C_2'],df_test['pred_a_23'])

    axs[2,2].scatter(df_test['komegasst_C_2'],df_test['DNS_a_33'])
    axs[2,2].scatter(df_test['komegasst_C_2'],df_test['pred_a_33'])

    fig.savefig(os.path.join(config.results_dir,f'{model.barcode}_fp_a_test.png'),dpi=300)

def add_predictions_to_df(gn, b_pred, b_perp_pred, a_pred, a_perp_pred, df_test):
    df_test[f'pred_b_11'] = b_pred[:,0,0]
    df_test[f'pred_b_13'] = b_pred[:,0,2]
    df_test[f'pred_b_22'] = b_pred[:,1,1]
    df_test[f'pred_b_12'] = b_pred[:,0,1]
    df_test[f'pred_b_23'] = b_pred[:,1,2]
    df_test[f'pred_b_33'] = b_pred[:,2,2]

    df_test[f'pred_a_11'] = a_pred[:,0,0]
    df_test[f'pred_a_12'] = a_pred[:,0,1]
    df_test[f'pred_a_13'] = a_pred[:,0,2]
    df_test[f'pred_a_22'] = a_pred[:,1,1]
    df_test[f'pred_a_23'] = a_pred[:,1,2]
    df_test[f'pred_a_33'] = a_pred[:,2,2]

    df_test[f'pred_g1'] = -np.ones(len(b_pred))
    df_test[f'pred_g2'] = gn[:,1]
    df_test[f'pred_g3'] = gn[:,2]
    df_test[f'pred_g4'] = gn[:,3]
    df_test[f'pred_g5'] = gn[:,4]
    df_test[f'pred_g6'] = gn[:,5]
    df_test[f'pred_g7'] = gn[:,6]
    df_test[f'pred_g8'] = gn[:,7]
    df_test[f'pred_g9'] = gn[:,8]
    df_test[f'pred_g10'] = gn[:,9]
    df_test[f'pred_logDelta'] = gn[:,10]

    return df_test



