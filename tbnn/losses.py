import torch

def bLoss(outputs, labels, alpha = 10):
    outputs = torch.nan_to_num(outputs)
    se = squaredError(outputs, labels)
    re = realizabilityLoss(outputs)
    return (se.mean()+alpha*re.mean())

def bLossPlus(outputs, labels, g_out, g1tilde_label, alpha = 10, gamma=0):
    bLoss_ = bLoss(outputs, labels, alpha)
    g1Loss = g1tildeLoss(g_out,g1tilde_label)
    return (bLoss_ + gamma*g1Loss)

def g1tildeLoss(g_out, g1tilde_label):
    se = (torch.log(-g_out[:,0]).reshape(-1,1) - g1tilde_label)**2
    return se.mean()

def squaredError(outputs, labels):
    se = ((outputs[:,0,0] - labels[:,0,0])**2 \
           + (outputs[:,0,1] - labels[:,0,1])**2 \
           + (outputs[:,0,2] - labels[:,0,2])**2 \
           + (outputs[:,1,1] - labels[:,1,1])**2 \
           + (outputs[:,1,2] - labels[:,1,2])**2 \
           + (outputs[:,2,2] - labels[:,2,2])**2 \
          )/6
    return se

def realizabilityLoss(outputs):
    re = realizabilityPenalty(outputs)
    return re.mean()

def mseLoss(outputs, labels):
    se = squaredError(outputs, labels)
    return se.mean()

def realizabilityPenalty(outputs):
    re_c = realizabilityPenalty_components(outputs)
    re_e = realizabilityPenalty_eigs(outputs)
    return re_c + re_e

def realizabilityPenalty_components(outputs):
    outputs = torch.nan_to_num(outputs)
    zero = torch.zeros_like(outputs)
    re_c = (torch.maximum(torch.maximum(outputs[:,0,0]-2/3, -(outputs[:,0,0] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,1]-2/3, -(outputs[:,1,1] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,2,2]-2/3, -(outputs[:,2,2] + 1/3)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,1]-1/2, -(outputs[:,0,1] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,0,2]-1/2, -(outputs[:,0,2] + 1/2)), zero[:,0,0])**2 \
        + torch.maximum(torch.maximum(outputs[:,1,2]-1/2, -(outputs[:,1,2] + 1/2)), zero[:,0,0])**2 \
          )/6 
    return re_c

def realizabilityPenalty_eigs(outputs):
    eigs = torch.sort(torch.real(torch.linalg.eigvals(outputs)),descending=True)[0]
    zero_eig = torch.zeros_like(eigs[:,0])
    re_eig1 = (torch.maximum((3*torch.abs(eigs[:,1])-eigs[:,1])/2 - eigs[:,0],zero_eig)**2)
    re_eig2 = (torch.maximum(eigs[:,0] - (1/3 - eigs[:,1]),zero_eig)**2)
    return (re_eig1 + re_eig2)