import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

pl.seed_everything(42)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudn


#Score: 0.28510 #
#input_size = 1  
#hidden_size = 128
#output_size = 1
class SimpleRNN_1(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size,24, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.sig(self.fc(out))
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
        self.fc1 = nn.Linear(hidden_size,500)
        self.fc2 = nn.Linear(500,output_size)
        #self.fc2 = nn.Linear(linear_size,output_size)
        self.sig=nn.Sigmoid()
        self.lr=learning_rate
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out= self.fc1(out)
        out = self.sig(self.fc2(out))
        
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

        self.model=BPPReactivityModel(input_channels=1, hidden_size=24, num_layers=12)

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


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion):
        super(InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expansion)
        
        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=500):
        super(MobileNetV2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            
            InvertedResidual(160, 320, 1, 6),
            
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
             nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create an instance of the MobileNetV2 model

class BPPReactivityPredictorWithMV2(pl.LightningModule):
    def __init__(self):
        super(BPPReactivityPredictorWithMV2, self).__init__()

        self.model=MobileNetV2()

        self.criterion = nn.MSELoss()#torch.nn.CrossEntropyLoss()

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

class SimpleNN(pl.LightningModule):
    def __init__(self,l_size=480, learning_rate=0.005):
        super(SimpleNN, self).__init__()
        self.learning_rate=learning_rate
        self.fc1 = nn.Linear(480,l_size)
        #self.fc2 = nn.Linear(480,480)
        #self.fc3 = nn.Linear(480,480)
        self.fc4 = nn.Linear(l_size,l_size)
        self.fc5 = nn.Linear(l_size,480)
        self.relu=nn.ReLU()
        self.sig=nn.Sigmoid()
        self.lr=learning_rate
        
    def forward(self, x):
        #print(x.shape)
        x=self.relu(self.fc1(x))
        #x=self.relu(self.fc2(x))
        #x=self.relu(self.fc3(x))
        x=self.relu(self.fc4(x))
        output=self.sig(self.fc5(x))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
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
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class CRNN(pl.LightningModule):
    def __init__(self,l_size=180, learning_rate=0.005):
        super(CRNN, self).__init__()
        self.learning_rate=learning_rate
        self.hidden_dim = l_size
        self.n_layers = 16
        self.transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12,batch_first=True)

        #self.rnn = nn.RNN(480,l_size,self.n_layers, batch_first=True)   
        self.fc1 = nn.Linear(l_size,l_size)
        #self.fc2 = nn.Linear(480,480)
        #self.fc3 = nn.Linear(480,480)
        self.fc4 = nn.Linear(l_size,l_size)
        self.fc5 = nn.Linear(l_size,480)
        self.relu=nn.ReLU()
        self.sig=nn.Sigmoid()
        self.lr=learning_rate
        
    def forward(self, x):
        #print(x.shape)
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        #hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out= self.transformer_model(x)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        x=self.relu(self.fc1(out))
        #x=self.relu(self.fc2(x))
        #x=self.relu(self.fc3(x))
        x=self.relu(self.fc4(x))
        output=self.sig(self.fc5(x))
        return output
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
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
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,batch_first=True)

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt=src
            output = self.transformer(src,tgt)
        else:
            output = self.transformer(src, tgt)
        return output


class SimpleTFModel(pl.LightningModule):
    def __init__(self,d_model, nhead, num_encoder_layers, num_decoder_layers,learning_rate=None):
        super(SimpleTFModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,batch_first=True,dropout=0.2)
        self.lr=learning_rate
        
    def forward(self, src, tgt=None):
        if tgt is None:
            tgt=src
            output = self.transformer(src,tgt)
        else:
            output = self.transformer(src, tgt)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
        loss = nn.MSELoss()(outputs.squeeze(), y.squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch[0]))
        x, y = batch
        #print(x.shape)
        outputs = self(x)  # Add an extra dimension for input_size
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
        # Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer