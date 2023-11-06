from abc import ABC, abstractmethod
from data_structures.step import Step

class BaseModel(ABC):
    @abstractmethod
    def select_action(state: int):
        pass

    @abstractmethod
    def step_update(step: Step):
        pass

    @abstractmethod
    def episode_update(step: Step):
        pass