import torch
import torch.nn as nn
from rnadatasets import *

class BPPReactivityModel(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        height, width = 177, 177  # Adjust this based on your actual input size

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

# Example usage
hidden_size = 64
bp = BPPReactivityModel(input_channels=1, hidden_size=hidden_size, num_layers=1)

random_input = torch.randn(1,1, 1, 177, 177)
#output = bp(random_input)

#print("Output shape:", output.shape)

print(len(StructureProbDatasetWithFixed500("../2A3_MaP")))

for x,y in StructureProbDatasetWithFixed500("../2A3_MaP"):
    x=x.unsqueeze(0)#.unsqueeze(0)
    
    count_above_threshold = (x[0][0] > 0.8).sum().item()
    print(count_above_threshold)
    #out=bp(x)
    #print(out)
    break