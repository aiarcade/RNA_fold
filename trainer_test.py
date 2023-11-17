import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import random_split
from torchvision import transforms
from rnadatasets import *
from settings import *

import pytorch_lightning as pl

class BPPReactivityPredictor(pl.LightningModule):
    def __init__(self, input_channels, output_size):
        super(BPPReactivityPredictor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Adjusted Fully connected layers
        self.fc1 = nn.Linear(64 * 177 * 177, 128)
        self.fc2 = nn.Linear(128, output_size)

        # Activation function
        self.relu = nn.ReLU()

        # Loss criterion
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Input size: (batch_size, input_channels, height, width)

        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        # Training step
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def configure_optimizers(self):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Example usage:
# Replace input_channels and output_size with appropriate values based on your data
input_channels = 1  # Assuming a single-channel BPP matrix
output_size = 177   # Assuming 177 reactivity values per input

# Instantiate the Lightning module
model = BPPReactivityPredictor(input_channels, output_size)

# Dummy data for example



datamodule = ProbDataModule(target_dirs[0],batch_size=TRAIN_BATCH_SIZE)

# Training loop using PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=5,accelerator="gpu",devices=[0,1])  # Set gpus=0 for CPU-only training
trainer.fit(model, datamodule=datamodule)
