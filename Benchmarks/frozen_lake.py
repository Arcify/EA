import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import envs

def frozen_lake_objective(environment, genome):
    total_reward = 0
    terminated = False
    state = environment.reset()
    while not terminated:
        actions = environment.action_space
        next_state, reward, terminated, info = environment.step(genome[state])
        total_reward += reward
        state = next_state
    return -total_reward