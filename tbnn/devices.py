import torch
def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f'Device: {device}')
    return device

#dev = get_device()
#print(dev)