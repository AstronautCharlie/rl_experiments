import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import math
import random

from models.base_model import BaseModel
from models.utils import Transition, ReplayMemory



class DQN(BaseModel):
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
                 lr=1e-4,
                 replay_memory_size=1000000,
                 optimizer=optim.AdamW):
        if len(hidden_layers) < 1:
            raise ValueError(f'Must have non-empty hidden layers :: got {hidden_layers}')
        
        class Network(nn.Module):
            """
            Densely connected NN with input/output/shape determined by 
            DQN inputs
            """
            def __init__(self):
                super(Network, self).__init__()
                self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
                for i in range(1, len(hidden_layers)):
                    self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.layers.append(nn.Linear(hidden_layers[-1], output_size))
            
            def forward(self, x):
                for layer in self.layers[:-1]:
                    x = F.relu(layer(x))
                # Don't apply relu to final layer 
                return self.layers[-1](x)
        
        self.policy_net = Network().to('cpu')
        self.target_net = Network().to('cpu')
        print(self.policy_net)
        print(self.policy_net.parameters())
        # Initialize policy and target networks to same parameters 
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(replay_memory_size)
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

        self.steps_taken = 0

    def episode_update(self, step):
        pass

    def step_update(self, step):
        pass

    def select_action(self, state):
        print(f'selection action for state {state}, {type(state)}')
        sample = random.random()
        if sample > self.get_epsilon_decayed(): 
            return self.get_best_action(state)
        else:
            return self.get_random_action()
        
    def get_best_action(self, state):
        print(f'getting best action')
        print(f'state before wrapping {state}')
        state = torch.tensor(state, dtype=torch.float32, device='cpu').unsqueeze(0)
        print(f'state after wrapping {state}')
        with torch.no_grad():
            print(f'policy net: {self.policy_net, state, self.policy_net(state)}')
            return self.policy_net(state).max(1)[1].view(1,1)

    def get_epsilon_decayed(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_taken / self.eps_decay)
    
    def get_random_action(self):
        print('getting random action')
        return torch.tensor([[random.randrange(self.output_size)]])