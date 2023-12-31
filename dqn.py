import gymnasium as gym 
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from tests.test_data import SimpleEnv

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO)

env = gym.make('CartPole-v1')
env = SimpleEnv()
logging.info(f'using env :: {env}')
n_actions = 2# env.action_space.n
state, info = env.reset()
n_observations = len(state)


is_ipython = 'inline' in matplotlib.get_backend()
logging.info(f'using iPython? :: {is_ipython}')

plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'using device :: {device}')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class ReplayMemory(object): 
    def __init__(self, capacity): 
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size): 
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions): 
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x): 
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
print(policy_net)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)

steps_done = 0 

def select_action(state): 
    global steps_done 
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1 
    if sample > eps_threshold: # exploitation
        with torch.no_grad():
            if steps_done == 10: 
                logging.info(f'state, policy_net(state) :: {state}, {policy_net(state)}')
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        #return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return torch.tensor([[random.randrange(0,2)]])
episode_durations = [] 

def plot_durations(show_result=False):
    plt.figure()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Trainig...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001) # pause to update plots
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model(): 
    if len(memory) < BATCH_SIZE:
        return 
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions)) # One Transition of concatenated states/actions/rewards

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    #logging.info(f'updating state/action/reward: {state_batch, action_batch, reward_batch}')
    #logging.info(f'policy net pre-update {policy_net(state_batch)}')

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    #logging.info(f'next_state_values shape :: {next_state_values.shape}')
    #logging.info(f'non_final_mask shape : {non_final_mask.shape}')
    #logging.info(f'target_net(non_final_next_states).shape : {target_net(non_final_next_states).shape}')
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    #logging.info(f'policy net post-update {policy_net(state_batch)}')


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 500

for i in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done: 
            episode_durations.append(t+1)
            print(f'finished episode {i} :: steps taken: {t}')
            #plot_durations()
            break

logging.info('policy learned:')
logging.info(f'state 0: {target_net(torch.tensor([0.]))}')
logging.info(f'state 1: {target_net(torch.tensor([1.]))}')
logging.info(f'state 2: {target_net(torch.tensor([2.]))}')


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()