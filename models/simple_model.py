import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class dumbnet(nn.Module):
    def __init__(self, state_size=1):
        super(dumbnet, self).__init__()
        self.layer1 = nn.Linear(state_size, 10)
        self.layer2 = nn.Linear(10,5)
        self.layer3 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    