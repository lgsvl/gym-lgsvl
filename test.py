#!/usr/bin/env python3
import gym
import numpy as np

env = gym.make('gym_lgsvl:lgsvl-v0')
env._setup_ego()

count = 6
while (count > 0):
    env._setup_npc()
    count -= 1

while (True):
    action = env.action_space.sample()
    env.step(action)