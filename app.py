import gymnasium as gym
import logging
from exp_box import ExperimentBox
from models.dqn import DQN
from telemetries import EpisodeLength
from tests.test_data import SimpleEnv

logging.basicConfig(level=logging.INFO)

env = gym.make('CartPole-v1')
env = SimpleEnv()
state_space_dim = 1
actions_available = 2
num_episodes=500

exp = ExperimentBox(model=DQN(input_size=state_space_dim, output_size=actions_available), env=env, telemetry=EpisodeLength(num_episodes=num_episodes))
exp.run_experiment()