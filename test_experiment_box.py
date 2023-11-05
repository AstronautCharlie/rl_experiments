from unittest import TestCase
from test_data import ToyEnv, ToyModel
from data_structures.telemetries import EpisodeLength
from params.params import SimpleParams
from exp_box import ExperimentBox
import pandas as pd
import os

class ExpBoxTest(TestCase):
    def test_happy_path(self):
        exp = ExperimentBox(model=ToyModel(), env=ToyEnv(), telemetry=EpisodeLength(), params=SimpleParams(num_episodes=10))
        exp.run_experiment()
        df = pd.read_csv('episode_lengths.csv')
        # Using toy environment, every episode should have length 4
        assert len(df[df['episode_lengths'] == 4]) == len(df)
        os.remove('episode_lengths.csv')

