import pandas as pd
from abc import ABC, abstractmethod
import logging

class BaseTelemetry(ABC):
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    @abstractmethod
    def step_update(self, step):
        pass

    @abstractmethod
    def episode_update(self, step_count: int, step_update):
        pass

    @abstractmethod
    def completion_conditions_met(self):
        pass

class EpisodeLength(BaseTelemetry):
    def __init__(self, *, file_location='results.csv', num_episodes=10):
        super(EpisodeLength, self).__init__()
        self.episode_lengths = []
        self.file_loc = file_location
        self.num_episodes_to_record = num_episodes

    def step_update(self, step):
        pass

    def episode_update(self, step_count: int, step_update):
        self.add_episode(step_count)

    def completion_conditions_met(self):
        return self.num_episodes_completed() == self.num_episodes_to_record
    
    def add_episode(self, ep_length: int):
        self.episode_lengths.append(ep_length)

    def write_results(self):
        df = pd.DataFrame({'episode_lengths': self.episode_lengths})
        df.to_csv(self.file_loc)

    def num_episodes_completed(self):
        return len(self.episode_lengths)