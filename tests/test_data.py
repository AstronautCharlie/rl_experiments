import torch
import numpy as np
import random

class ToyEnv:
    def __init__(self):
        self.state = torch.tensor([0])

    def reset(self):
        self.state = torch.tensor([0])
        return self.state, None

    def step(self, action): # all actions lead to state 0
        self.state += 1 
        terminated = False if self.state < 5 else True
        reward = 0 if self.state < 5 else 1
        print(f'returning state {self.state}, {type(self.state)}')
        return self.state, reward, terminated, False, None

class ToyModel:
    def select_action(self, state): # always take default action
        return torch.tensor([0])
    
    def step_update(self, step):
        pass

    def episode_update(self, step):
        pass

class SimpleEnv:
    """
    3-state, 2 action env. States a, b, c - actions 0,1
    Transition rules: 
        (a,0) -> a p=1
        (a,1) -> b p=1
        (b,0) -> b p=1 
        (b,1) -> c p=1
        (c,0) -> b p=0.75, game ends, c p=0.25
        (c,1) -> c p=0.75, a p=0.25
    Reward rules:
        a -> a: 0
        b -> b: 0
        c -> c: 0
        a -> b: 0
        b -> a: 0
        b -> c: 1
        c -> b: 1
        c -> a: -2
    Agent should learn the following policy: 
        a: 1
        b: 1
        c: 0
    """
    def reset(self):
        self.state = np.array([0.])
        return self.state, None
    
    def step(self, action): 
        if self.state == np.array([0.]):
            if action == 0:
                self.state = np.array([0.])
                # a -> a
                return np.array([0.], dtype=float), 0, False, False, None
            else:
                # a -> b
                self.state = np.array([1.])
                return np.array([1.], dtype=float), 0, False, False, None
        elif self.state == np.array([1.]):
            if action == 0:
                self.state = np.array([1.])
                # b -> b
                return np.array([1.], dtype=float), 0, False, False, None
            else:
                # b -> c 
                self.state = np.array([2.])
                return np.array([2.], dtype=float), 1, False, False, None
        else:
            if action == 0: 
                if random.random() < 0.75:
                    # c -> b
                    self.state = np.array([1.])
                    return np.array([1.], dtype=float), 1, True, False, None
                else:
                    # c -> c
                    return np.array([2.], dtype=float), 0, False, False, None
            else:
                if random.random() < 0.75: 
                    # c -> c
                    return np.array([2.], dtype=float), 0, False, False, None
                else:
                    self.state = np.array([0.])
                    return np.array([0.], dtype=float), -2, False, False, None