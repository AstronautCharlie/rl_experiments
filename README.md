# What is this? 
This is a framework for running RL experiments. An experiment consists of:
- a model to test 
- an environment
- a telemetry definition to record data
- experiment parameters

# How do I run this? 
`python app.py`

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
`models/dqn.py` runs but isn't learning - I suspect it's not picking the best action or something. Create a simple environment that does have best actions to take, and see what the model is learning there. Or just troubleshoot more. I think the squeeze in line 105 of `models/dqn.py` to make dimensions fit probably fucked things up. 