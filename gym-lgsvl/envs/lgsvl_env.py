import gym
from gym import error, spaces, utils
from gym.utils import seeding

class LgsvlEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        NotImplementedError

    def step(self, action):
        NotImplementedError

    def reset(self):
        NotImplementedError

    def render(self, mode='human'):
        NotImplementedError
    
    def close(self):
        NotImplementedError