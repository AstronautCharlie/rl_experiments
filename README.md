# What is this? 
This is a framework for running RL experiments. An experiment consists of:
- a model to test 
- an environment
- a telemetry definition to record data
- experiment parameters

## Models
Implementations of learning algorithms. Must implement the following: 
- `select_action(state)`
- `step_update(step: Step)` 
- `episode_update(step_count, step: Step)`

## Environments
Must implement the following:
- `step(action)`
- `reset()`

## Telemetries
Must implement the following:
- `step_update(step: Step)`
- `episode_update(step_count, step: Step)`

## Experiment Parameters
Must implement the following:
- `completion_conditions_met(telemetry)`

## To Test
`python -m pytest . -vv`

# Last Update
part-way through moving `dqn.py` into actual class in `models/dqn.py`