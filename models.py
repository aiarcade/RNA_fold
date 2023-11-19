import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

pl.seed_everything(42)



#Score: 0.28510 #
#input_size = 1  
#hidden_size = 128
#output_size = 1
class SimpleRNN_1(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,24, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        self.log("val_loss", loss, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())

        # Assuming a threshold of 0.5 for binary classification
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        correct_predictions = (predicted_labels == y.squeeze()).float()
        accuracy = correct_predictions.mean()
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': accuracy}


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

class SimpleRNN2(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=206, output_size=1,n_rnn_layers=16,linear_size=206,learning_rate=0.01):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size,n_rnn_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,linear_size)
        self.fc2 = nn.Linear(linear_size,output_size)
        self.lr=learning_rate
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        self.log("hp_metric",loss, on_step=False, on_epoch=True,sync_dist=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())

        # Assuming a threshold of 0.5 for binary classification
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        correct_predictions = (predicted_labels == y.squeeze()).float()
        accuracy = correct_predictions.mean()
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': accuracy}


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class SimpleRNN3(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=206, output_size=1,n_rnn_layers=16,linear_size=206,learning_rate=0.01):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,n_rnn_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)
        #self.fc2 = nn.Linear(linear_size,output_size)
        self.lr=learning_rate
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        self.log("hp_metric",loss, on_step=False, on_epoch=True,sync_dist=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())

        # Assuming a threshold of 0.5 for binary classification
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        correct_predictions = (predicted_labels == y.squeeze()).float()
        accuracy = correct_predictions.mean()
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': accuracy}


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class SimpleRNN(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=206, output_size=1,n_rnn_layers=16,linear_size=206,learning_rate=0.01):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,n_rnn_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size,output_size)
        #self.fc2 = nn.Linear(linear_size,output_size)
        self.lr=learning_rate
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc1(out)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        self.log("hp_metric",loss, on_step=False, on_epoch=True,sync_dist=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x.unsqueeze(-1))  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())

        # Assuming a threshold of 0.5 for binary classification
        predicted_labels = (outputs.squeeze() >= 0.5).float()
        correct_predictions = (predicted_labels == y.squeeze()).float()
        accuracy = correct_predictions.mean()
        self.log('test_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': accuracy}


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

class BPPReactivityPredictor(pl.LightningModule):
    def __init__(self, input_channels, output_size):
        super(BPPReactivityPredictor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Adjusted Fully connected layers
        self.fc1 = nn.Linear(64 * output_size * output_size, 128)
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
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class BPPReactivityModel(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        height, width = 500, 500  # Adjust this based on your actual input size

        self.rnn = nn.RNN(input_size=height * width, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Global Average Pooling
        self.fc2 = nn.Linear(hidden_size,500)  # Use hidden_size as the input size for the linear layer
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # Loss criterion
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Reshape the tensor before passing it to the RNN layer
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)

        # Assuming you want to use the output of the last time step from the RNN
        _, hn = self.rnn(x)
        x = self.relu(hn[-1])  # Using the output of the last time step

        output = self.sigmoid(self.fc2(x))  # Directly pass the RNN output to the linear layer

        return output


class BPPReactivityPredictorWithRNN(pl.LightningModule):
    def __init__(self):
        super(BPPReactivityPredictorWithRNN, self).__init__()

        self.model=BPPReactivityModel(input_channels=1, hidden_size=32, num_layers=32)

        self.criterion = nn.MSELoss()

    def forward(self,x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        # Training step
        inputs, targets = batch
        
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        return loss
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        #inputs=inputs.squeeze(0)
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, prog_bar=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
