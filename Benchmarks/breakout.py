

import numpy as np



def breakout_objective(env, genome, model):
    reshaped_genome = [[i] for i in genome]
    model.layers[1].set_weights([np.array(reshaped_genome), np.array([0])])
    total_reward = 0
    terminated = False
    state = env.reset()
    while not terminated:
        #s_new = np.expand_dims(state, 0)
        #env.render()
        s_new = state.reshape(-1)
        s_new = s_new.astype('float64')
        #very inefficient, need better way of doing this!!!
        observations = [[i] for i in s_new]
        prediction = model.predict([observations]) #outputs very large numbers, need to fix this
        print(prediction)
        action = get_action(prediction)
        next_state, reward, terminated, info = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

def get_action(prediction):
    if 0 < prediction < 0.25:
        return 0
    elif 0.25 < prediction < 0.5:
        return 1
    elif 0.5 < prediction < 0.75:
        return 2
    else:
        return 3