import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import logging

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
                 optimizer=optim.AdamW,
                 criterion=nn.SmoothL1Loss):
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
        self.criterion = criterion

        self.steps_taken = 0

    def episode_update(self, step):
        pass

    def step_update(self, step):
        self.steps_taken += 1 
        self.memory.push(step.state, step.action, step.next_state, step.reward)

        self.optimize_policy_net(step)

        self.update_target_net()

    def optimize_policy_net(self, step):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions)) # One Transition of concatenated states/actions/rewards

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # MxN shape tensor where m is batch size and n is state dimensionality
        state_batch = torch.cat(batch.state)
        # Mx1 shape tensor where m is batch size and 1 b/c action number
        action_batch = torch.cat(batch.action).unsqueeze(1)
        # Mx1 shape tensor
        reward_batch = torch.cat(batch.reward)

        # Get the value of each sampled state/action pair, according to policy net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

        next_state_values = torch.zeros(self.batch_size, device='cpu')
        #logging.info(f'next_state_values shape :: {next_state_values.shape}')
        #logging.info(f'non_final_mask shape : {non_final_mask.shape}')
        #logging.info(f'target_net(non_final_next_states).shape : {self.target_net(non_final_next_states).shape}')

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).squeeze(1).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = self.criterion()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def select_action(self, state):
        sample = random.random()
        if sample > self.get_epsilon_decayed(): 
            return self.get_best_action(state)
        else:
            return self.get_random_action()
        
    def get_best_action(self, state):
        state = torch.tensor(state, device='cpu').unsqueeze(0)
        with torch.no_grad():
            #logging.info(f'picking best action') 
            #logging.info(f'action values: {self.policy_net(state)}')
            #logging.info(f'picking action {self.policy_net(state).max(1)[1]}')
            return self.policy_net(state).max(1)[1].view(1,1).item()

    def get_epsilon_decayed(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_taken / self.eps_decay)
    
    def get_random_action(self):
        return torch.tensor([[random.randrange(self.output_size)]]).item()