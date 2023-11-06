import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

from models.base_model import BaseModel
from models.utils import Transition, ReplayMemory

class DQN(BaseModel, nn.Module):
    def __init__(self, *, 
                 hidden_layers=[10],
                 input_size=2,
                 output_size=1,
                 batch_size=128, 
                 gamma=0.99,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=1000,
                 tau=0.005,
                 lr=1e-4):
        nn.Module.__init__(self)
        if len(hidden_layers) < 1:
            raise ValueError(f'Must have non-empty hidden layers :: got {hidden_layers}')
        self.hidden_layers = hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.initialize_networks()

    def initialize_networks(self):
        
    
    def _initialize_network(self):
        network = {0: nn.Linear(self.input_size, self.hidden_layers[0])}

        for i in range(1, len(self.hidden_layers)):
            network[i] = nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i])

        network[i+1] = nn.Linear(self.hidden_layers[i], self.output_size)
    
    def forward(self, x):
        for _, layer in self.network.items():
            x = F.relu(layer(x))
        return x 
        
    def episode_update(self, step):
        pass

    def step_update(self, step):
        pass

    def select_action(self, state: int):
        return 0
    