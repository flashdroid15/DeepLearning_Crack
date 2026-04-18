import torch.nn as nn

class CrackAutoencoder(nn.Module):
    def __init__(self):
        super(CrackAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            # Layer 1: 224x224 -> 112x112
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 112x112 -> 56x56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 56x56 -> 28x28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 28x28 -> 14x14
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Layer 5 (Bottleneck): 14x14 -> 7x7
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(512 * 7 * 7, 128)

        self.fc_up = nn.Linear(128, 512 * 7 * 7)
        self.unflatten = nn.Unflatten(1, (512, 7, 7))

        self.decoder = nn.Sequential(
            # Layer 1: 7x7 -> 14x14
            # output_padding=1 ensures the dimensions scale back up exactly by 2x
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Layer 2: 14x14 -> 28x28
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 3: 28x28 -> 56x56
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 4: 56x56 -> 112x112
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final Layer: 112x112 -> 224x224 (Back to RGB)
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.bottleneck(x)
        
        x = self.fc_up(latent)
        x = self.unflatten(x)
        reconstruction = self.decoder(x)
        return reconstruction


class LightweightCrackAutoencoder(nn.Module):
    def __init__(self):
        super(LightweightCrackAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(128 * 14 * 14, 128)

        self.fc_up = nn.Linear(128, 128 * 14 * 14)
        self.unflatten = nn.Unflatten(1, (128, 14, 14))

        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Final Layer: 112x112 -> 224x224 (Back to RGB)
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.bottleneck(x)    
        
        x = self.fc_up(latent)
        x = self.unflatten(x)
        return self.decoder(x)

class HeavyweightCrackAutoencoder(nn.Module):
    def __init__(self):
        super(HeavyweightCrackAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: 224x224 -> 112x112
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Layer 5: 14x14 -> 7x7
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )
        
        self.flatten = nn.Flatten()
        
        self.bottleneck = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3) # 30% Dropout prevents perfect pixel memorization
        )

        self.fc_up = nn.Sequential(
            nn.Linear(1024, 1024 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.unflatten = nn.Unflatten(1, (1024, 7, 7))

        self.decoder = nn.Sequential(
            # Layer 1: 7x7 -> 14x14
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Layer 2: 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Layer 3: 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 4: 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final Layer: 112x112 -> 224x224 (Back to RGB)
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        latent = self.bottleneck(x)    
        
        x = self.fc_up(latent)
        x = self.unflatten(x)
        return self.decoder(x)