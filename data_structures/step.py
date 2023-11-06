class Step:
    def __init__(self, action, observation, reward, terminated, truncated, info): # Expected in order from Gymnasium==0.29.1
        self.action = action
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def ends_episode(self):
        return self.truncated or self.terminated