#!/usr/bin/env python3
import gym
import numpy as np

env = gym.make('gym_lgsvl:lgsvl-v0')
observation = env.reset()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode return = {}".format(env.reward))
            break
