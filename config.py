import torch
import torchvision.transforms as transforms

# Returns the device to be used for training
def get_device():
    mps_device = torch.device("cpu")

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    elif torch.cuda.is_available():
        mps_device = torch.device("cuda")
    
    print("Using device:", mps_device)
    return mps_device

# Torch hyperparameters
transform = transforms.Compose([
    transforms.ToTensor(),
])
num_workers = 6

# Experiment data
LEFT_POS = 105
RIGHT_POS = 151

slash_position = {
    "start": 50,
    "end": 225,
}

