import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 12


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 75 * 50, 256)
        self.fc2 = nn.Linear(256, 3)
        # Initialize weights
        self.init_weights()

    def forward(self, x):
        # Apply first convolution and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second convolution and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output
        x = x.view(-1, 16 * 75 * 50)
        # Apply first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer
        x = self.fc2(x)
        return x

    def init_weights(self):
        torch.manual_seed(seed)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
