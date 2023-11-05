class ToyEnv:
    def __init__(self):
        self.state = 0 

    def reset(self):
        self.state = 0
        return 0, None

    def step(self, action): # all actions lead to state 0
        self.state += 1 
        terminated = False if self.state < 5 else True
        reward = 0 if self.state < 5 else 1
        return self.state, reward, terminated, False, None

class ToyModel:
    def select_action(self, state): # always take default action
        return 0
    
    def step_update(self, step):
        pass

    def episode_update(self, step):
        pass