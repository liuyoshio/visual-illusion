import os
import torch
import torch.nn as nn

MODEL_PATH = '../models/unet_default.pth'

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class Unet(nn.Module):
    def __init__(self, in_channels=3, features=64, model_path=MODEL_PATH):
        super().__init__()
        # 256x256x3
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        # 128x128x64
        self.down1 = Block(features, features*2, down=True, act='leaky', use_dropout=False)
        # 64x64x128
        self.down2 = Block(features*2, features*4, down=True, act='leaky', use_dropout=False)
        # 32x32x256
        self.down3 = Block(features*4, features*8, down=True, act='leaky', use_dropout=False)
        # 16x16x512
        self.down4 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        # 8x8x512
        self.down5 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        # 4x4x512
        self.down6 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        # 2x2x512

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU(),
        )
        # 1x1x512

        self.up1 = Block(features*8, features*8, down=False, act='leaky', use_dropout=True)
        # 2x2x1024
        self.up2 = Block(features*8*2, features*8, down=False, act='leaky', use_dropout=True)
        # 4x4x1024
        self.up3 = Block(features*8*2, features*8, down=False, act='leaky', use_dropout=True)
        # 8x8x1024
        self.up4 = Block(features*8*2, features*8, down=False, act='leaky', use_dropout=False)
        # 16x16x1024
        self.up5 = Block(features*8*2, features*4, down=False, act='leaky', use_dropout=False)
        # 32x32x512
        self.up6 = Block(features*4*2, features*2, down=False, act='leaky', use_dropout=False)
        # 64x64x256
        self.up7 = Block(features*2*2, features, down=False, act='leaky', use_dropout=False)
        # 128x128x128

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, 2, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        # 256x256x2
        self.model_path = model_path

        if os.path.exists(model_path):
            self.load()

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        return self.final_up(torch.cat([u7, d1], dim=1))
    
    def save(self, path=MODEL_PATH):
        torch.save(self.state_dict(), path)
    
    def load(self, path=MODEL_PATH):
        print('Load model from', path)
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None))
    
    # get the base name of the model path
    # eg: '../models/unet_default.pth' -> 'unet_default'
    def get_network_dir(self):
        return os.path.splitext(os.path.basename(self.model_path))[0]

def get_model(in_channels=3, features=64, model_path=MODEL_PATH):
    return Unet(in_channels=in_channels, features=features, model_path=model_path)