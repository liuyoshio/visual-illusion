import torch
import torchvision.transforms as transforms

# Returns the device to be used for training
def get_device():
    mps_device = torch.device("cpu")

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    elif torch.backends.cuda.is_available():
        mps_device = torch.device("cuda")
    
    print("Using device:", mps_device)
    return mps_device

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Experiment data

LEFT_POS = 0



