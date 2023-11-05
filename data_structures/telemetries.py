import pandas as pd
from abc import ABC, abstractmethod

class Step:
    def __init__(self, action, observation, reward, terminated, truncated, info): # Expected in order from Gymnasium==0.29.1
        self.action = action
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def ends_episode(self):
        return self.truncated or self.terminated

class BaseTelemetry(ABC):
    @abstractmethod
    def step_update(self, step):
        pass

    @abstractmethod
    def episode_update(self, step_count: int, step_update):
        pass

class EpisodeLength(BaseTelemetry):
    def __init__(self, file_location=None):
        self.episode_lengths = []
        self.file_loc = file_location

    def step_update(self, step):
        pass

    def episode_update(self, step_count: int, step_update):
        self.add_episode(step_count)
    
    def add_episode(self, ep_length: int):
        self.episode_lengths.append(ep_length)

    def write_results(self):
        file_loc = self.file_loc if self.file_loc is not None else 'episode_lengths.csv'
        df = pd.DataFrame({'episode_lengths': self.episode_lengths})
        df.to_csv(file_loc)

    def num_episodes_completed(self):
        return len(self.episode_lengths)