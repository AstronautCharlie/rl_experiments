from exp_box import ExperimentBox
from models.dqn import DQN
from telemetries import EpisodeLength
from test_data import ToyEnv, ToyModel

exp = ExperimentBox(model=DQN(), env=ToyEnv(), telemetry=EpisodeLength())
exp.run_experiment()