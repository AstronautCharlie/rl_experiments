
from itertools import count
from params.params import BaseParams 
from data_structures.telemetries import Step
from settings import ExpConfig as Config

class ExperimentBox:
    def __init__(self, *, model, env, telemetry, params: BaseParams):
        self.model = model
        self.env = env
        self.params = params
        self.telemetry = telemetry

    def run_experiment(self):
        self.experiment_complete = False
        while not self.experiment_is_over():
            self.run_episode()
        self.telemetry.write_results()

    def experiment_is_over(self):
        return self.params.completion_conditions_met(self.telemetry)

    def run_episode(self):
        state, _ = self.env.reset()
        for t in count():
            # Get action
            action = self.model.select_action(state)
            # Apply action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # Bundle all the step info together
            step = Step(action, next_state, reward, terminated, truncated, info)
            # Make step update to model
            self.step_update(step)
            # if episode is over, make episode update
            if step.ends_episode():
                self.episode_update(t, step)
                return
            # otherwise, set state to next_state
            state = next_state

    def step_update(self, step: Step):
        self.model.step_update(step)
        self.telemetry.step_update(step)

    def episode_update(self, step_count: int, step_update: Step):
        self.model.episode_update(step_update)
        self.telemetry.episode_update(step_count, step_update)