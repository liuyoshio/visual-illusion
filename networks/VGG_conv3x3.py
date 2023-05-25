import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

MODEL_PATH = '../model/vgg_conv3x3.pth'

class VGG_conv3x3(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 64x64x1
        self.activation = nn.Tanh()
        self.conv = nn.Conv2d(in_channels=64, out_channels=2, 
                              kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        
        if os.path.exists(MODEL_PATH):
            self.load(MODEL_PATH)
            print('Load model from', MODEL_PATH)
        
    def forward(self, x):
        feature = self.model(x)
        label = self.conv(feature)
        label = self.tanh(label)
        return label
    
    def save(self):
        torch.save(self.state_dict(), MODEL_PATH)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

def get_model():
    model = torchvision.models.vgg16(pretrained=True)
    model = nn.Sequential(*list(model.features.children())[:4])
    decoder = VGG_conv3x3(model)
    for param in decoder.model.parameters():
        param.requires_grad = False
    return decoder