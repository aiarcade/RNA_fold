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