import torch
import torch.nn as nn

class CNN_XL(nn.Module):

    def __init__(self, num_class=10):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            
            # block 1: 3x32x32 -> 64x16x16
            nn.Conv2d(3,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # block 2: 64x16x16 -> 128x8x8
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # block 3: 128x8x8 -> 256x4x4
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # flatten the final conv block
        self.flatten_dim = 256 * 4 * 4

        # FC Layer
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
         
         # complete conv here
         x = self.conv_blocks(x)

         # flatten the conv output
         x = x.view(x.size(0), -1)

         # run the classifier
         x = self.classifier(x)

         return x