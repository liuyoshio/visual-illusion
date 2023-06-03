import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

MODEL_PATH = '../model/vgg_conv3x3.pth'

class VGG_conv3x3(nn.Module):
    def __init__(self, model, model_path):
        super().__init__()
        self.model = model
        self.model_path = model_path
        # 64x64x1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, 
                              kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2= nn.Conv2d(in_channels=128, out_channels=2, 
                              kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.Tanh()
        
        if os.path.exists(self.model_path):
            self.load(self.model_path)
            print('Load model from', self.model_path)
        
    def forward(self, x):
        label = self.model(x)
        label = self.conv1(label)
        label = self.activation1(label)
        label = self.conv2(label)
        label = self.activation2(label)
        return label
    
    def save(self):
        torch.save(self.state_dict(), self.model_path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def get_network_dir(self):
        return os.path.splitext(os.path.basename(self.model_path))[0]

def get_model(model_path=MODEL_PATH):
    model = torchvision.models.vgg16(pretrained=True)
    model = nn.Sequential(*list(model.features.children())[:4])
    decoder = VGG_conv3x3(model, model_path)
    return decoder

def if_exist(path):
    import os
    return os.path.exists(path)