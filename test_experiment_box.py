from unittest import TestCase
from test_data import ToyEnv, ToyModel
from telemetries import EpisodeLength
from exp_box import ExperimentBox
import pandas as pd
import os

class ExpBoxTest(TestCase):
    def test_happy_path(self):
        exp = ExperimentBox(model=ToyModel(), env=ToyEnv(), telemetry=EpisodeLength())
        exp.run_experiment()
        df = pd.read_csv('results.csv')
        # Using toy environment, every episode should have length 4
        assert len(df[df['episode_lengths'] == 4]) == len(df)
        os.remove('results.csv')

