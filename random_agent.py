#!/usr/bin/env python3
import gym
import numpy as np
from spinup.utils.logx import EpochLogger

env = gym.make('gym_lgsvl:lgsvl-v0')

observation = env.reset()
epoch_logger = EpochLogger()

for i_episode in range(20):
  observation = env.reset()
  while (True):   
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    epoch_logger.store(Reward = reward)
    if done:
        epoch_logger.log_tabular('Reward', with_min_and_max=True)
        epoch_logger.dump_tabular()
        break
