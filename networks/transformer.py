import torch
import torch.nn as nn
import os

MODEL_PATH = '../models/vit_default.pth'

class TransformerUNet(nn.Module):

    def __init__(self, in_channels, out_channels, model_path=MODEL_PATH):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        self.transformer = nn.Transformer(
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

        self.model_path = model_path

    def forward(self, x):
        x = self.encoder1(x)
        batch_size, _, height, width = x.size()
        x = x.permute(0, 2, 3, 1).view(batch_size, -1, 256)  # Reshape for transformer
        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2).view(batch_size, height, width, 256).permute(0, 3, 1, 2)  # Reshape for decoder
        x = self.decoder1(x)
        return x
    
    def save(self):
        torch.save(self.state_dict(), self.model_path)
        print('Model saved to', self.model_path)
    
    def load(self):
        print('Load model from', self.model_path)
        self.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu') if not torch.cuda.is_available() else None))
    
    # get the base name of the model path
    # eg: '../models/unet_default.pth' -> 'unet_default'
    def get_network_dir(self):
        return os.path.splitext(os.path.basename(self.model_path))[0]


def get_model(model_path=MODEL_PATH):
    model = TransformerUNet(in_channels=3, out_channels=2, model_path=model_path)
    if os.path.exists(model_path):
        model.load()
    return model

if __name__ == '__main__':
    # Instantiate the model
    model = get_model()
    model.save()
    # Generate sample input tensor
    batch_size = 4
    input_channels = 3
    input_height = 256
    input_width = 256
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)

    # Pass the input through the model
    output_tensor = model(input_tensor)

    # Print the shape of the output tensor
    print("Output shape:", output_tensor.shape)
