import torch

def realizabilityLoss(b):
    re = realizabilityPenalty(b)
    return re.mean()

def mseLoss(outputs, labels):
    se = squaredError(outputs, labels)
    return se.mean()

def aLoss(b_pred, g_pred, k, a_label, amagmean, alpha = 1):
    se = mseLoss_aLoss(b_pred, g_pred, k, a_label, amagmean)
    re = ((1/torch.square(amagmean))*torch.square(2*k)*2*realizabilityPenalty(b_pred)).mean()
    return se + alpha*re

def mseLoss_aLoss(b_pred, g_pred, k, a_label, amagmean):
    se = (1/(torch.square(amagmean))).reshape(-1,1,1) * squaredError(2*k.reshape(-1,1,1)*b_pred, a_label)
    return se.mean()

def mse_a(b_pred, g_pred, k, a_label, amagmean):
    se = squaredError(2*k.reshape(-1,1,1)*b_pred, a_label)
    return se.mean()

def bLoss(outputs, labels, alpha = 1):
    outputs = torch.nan_to_num(outputs)
    se = squaredError(outputs, labels)
    re = realizabilityLoss(outputs)
    return (se.mean()+alpha*re.mean())

def squaredError(outputs, labels):
    se = ((outputs[:,0,0] - labels[:,0,0])**2 \
           + (outputs[:,0,1] - labels[:,0,1])**2 \
           + (outputs[:,0,2] - labels[:,0,2])**2 \
           + (outputs[:,1,1] - labels[:,1,1])**2 \
           + (outputs[:,1,2] - labels[:,1,2])**2 \
           + (outputs[:,2,2] - labels[:,2,2])**2 \
          )/6
    return se

def mse_k(k_pred, _, label):
    se = (k_pred - label)**2 
    return se.mean()

def mse_Delta(Delta_pred, label):
    se = (Delta_pred - label.view(-1,1))**2 
    return se.mean()

def realizabilityPenalty(b):
    re_c = realizabilityPenalty_components(b)
    re_e = realizabilityPenalty_eigs(b)
    return re_c + re_e

def realizabilityPenalty_components(b):
    b = torch.nan_to_num(b)
    zero = torch.zeros_like(b)
    re_c = (torch.maximum(torch.maximum(b[:,0,0]-2/3, -(b[:,0,0] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,1,1]-2/3, -(b[:,1,1] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,2,2]-2/3, -(b[:,2,2] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,0,1]-1/2, -(b[:,0,1] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,0,2]-1/2, -(b[:,0,2] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(b[:,1,2]-1/2, -(b[:,1,2] + 1/2)), zero[:,0,0])**2 \
          )/6 
    return re_c

def realizabilityPenalty_eigs(b):
    eigs = torch.sort(torch.real(torch.linalg.eigvals(b)),descending=True)[0]
    zero_eig = torch.zeros_like(eigs[:,0])
    re_eig1 = (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2)
    re_eig2 = (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    return (re_eig1 + re_eig2)/2

"""
def mseLoss_a_kmean(b_pred, k, kmean, a_label):
    a_pred = 2*k.reshape(-1,1,1)*b_pred
    se = squaredError(a_pred, a_label)
    se = torch.div(se,torch.square(2*kmean))
    return se.mean()
"""