from abc import ABC, abstractmethod
from data_structures.telemetries import BaseTelemetry

class BaseParams(ABC):
    @abstractmethod
    def completion_conditions_met(self, telemetry: BaseTelemetry):
        pass

class SimpleParams(BaseParams):
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes

    def completion_conditions_met(self, telemetry: BaseTelemetry):
        return telemetry.num_episodes_completed() == self.num_episodes