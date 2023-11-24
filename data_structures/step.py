import torch

class Step:
    def __init__(self, state, action, next_state, reward, terminated, truncated, info): # Expected in order from Gymnasium==0.29.1
        self.state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        self.next_state = torch.tensor([next_state], dtype=torch.float32).unsqueeze(0)
        self.reward = torch.tensor([reward], dtype=torch.float32)
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def ends_episode(self):
        return self.truncated or self.terminated