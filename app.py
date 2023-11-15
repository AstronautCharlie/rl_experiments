from exp_box import ExperimentBox
from models.dqn import DQN
from telemetries import EpisodeLength
from tests.test_data import ToyEnv, ToyModel

exp = ExperimentBox(model=DQN(input_size=1, output_size=2), env=ToyEnv(), telemetry=EpisodeLength())
exp.run_experiment()