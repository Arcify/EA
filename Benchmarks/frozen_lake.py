import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import envs
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


def frozen_lake_objective(environment, genome, model):
    reshaped_genome = [[i] for i in genome]
    model.layers[1].set_weights([np.array(reshaped_genome), np.array([0])])
    total_reward = 0
    terminated = False
    state = environment.reset()
    states_survived = 0
    while not terminated:
        observations = get_observation_space(state)
        prediction = model.predict([observations])
        action = get_action(prediction)
        next_state, reward, terminated, info = environment.step(action)
        total_reward += reward
        if next_state <= state:
            terminated = True
        state = next_state
        states_survived += 1
    print(-total_reward * 50 - states_survived)
    return -total_reward * 50 - states_survived

def get_observation_space(observation):
    observations = []
    for i in range(16):
        if i != observation:
            observations.append([0])
        else:
            observations.append([1])
    return observations

def get_action(prediction):
    if 0 < prediction < 0.25:
        return 0
    elif 0.25 < prediction < 0.5:
        return 1
    elif 0.5 < prediction < 0.75:
        return 2
    else:
        return 3
