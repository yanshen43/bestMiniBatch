import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn import init

# Basic CNN from PyTorch Tutorial
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d( 3, 6, 5 )
        self.pool = nn.MaxPool2d( 2, 2 )
        self.conv2 = nn.Conv2d( 6, 16, 5 )
        self.fc1 = nn.Linear( 16 * 5 * 5, 120 )
        self.fc2 = nn.Linear( 120, 84 )
        self.fc3 = nn.Linear( 84, 10 )

    def init_weight(self):
        # Initialize linear transform here
        pass

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
