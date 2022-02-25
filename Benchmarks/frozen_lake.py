import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import envs
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


def frozen_lake_objective(environment, genome):
    input = Input(shape=(16,))
    output = Dense(1, activation = 'sigmoid')(input)
    model = Model(inputs = input, outputs = output)
    model.compile(loss='mse', optimizer='adam')
    #model.layers[1].set_weights(genome)
    total_reward = 0
    terminated = False
    state = environment.reset()
    while not terminated:
        observations = get_observation_space(state)
        prediction = model.predict([observations])
        action = get_action(prediction)
        next_state, reward, terminated, info = environment.step(action)
        total_reward += reward
        state = next_state
    return -total_reward

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
