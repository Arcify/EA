
import gym
import numpy as np
from gym import envs
from Benchmarks.frozen_lake import frozen_lake_objective, run_env
from Benchmarks.breakout import breakout_objective
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense


class Evolutionary:

    #CITE: this algorithm has been implemented with help of the tutorial available here:
    # https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    def genetic_algorithm(self, objective, n_bits, pop_size, n_iter, r_cross, r_mut, env):
        #model = self.frozen_lake_neural_network()
        model = self.breakout_neural_network()
        print(model.layers[1].get_weights())
        population = [np.random.rand(n_bits).tolist() for _ in range(pop_size)]
        best, best_eval = 0, objective(env, population[0], model)
        for generation in range(n_iter):
            print(generation)
            scores = [objective(env, c, model) for c in population]
            for i in range(pop_size):
                if scores[i] < best_eval:
                    best, best_eval = population[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (generation, population[i], scores[i]))
            selected = [self.selection(population, scores) for _ in range(pop_size)]
            children = list()
            for i in range(0, pop_size, 2):
                p1, p2 = selected[i], selected[i + 1]
                for c in self.crossover(p1, p2, r_cross):
                    self.mutation(c, r_mut)
                    children.append(c)
            population = children
        return [best, best_eval]

    def selection(self, population, scores, k=2):
        selected = np.random.randint(0, len(population))
        for c in np.random.randint(0, len(population), k):
            if scores[c] < scores[selected]:
                selected = c
        return population[selected]

    def crossover(self, p1, p2, r_cross):
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross:
            pt = np.random.randint(1, len(p1) - 2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def mutation(self, bitstring, r_mut):
        for i in range(len(bitstring)):
            if np.random.rand() < r_mut:
                bitstring[i] = np.random.rand()

    def frozen_lake_neural_network(self):
        input = Input(shape=(16,))
        output = Dense(1, activation='linear')(input)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='mse', optimizer='adam')
        return model

    def breakout_neural_network(self):
        input = Input(shape=(100800,))
        output = Dense(1, activation='linear')(input)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='mse', optimizer='adam')
        return model

if __name__ == '__main__':
    # print(envs.registry.all())
    #env = gym.make('FrozenLake-v1', is_slippery=False)
    algorithm = Evolutionary()
    env = gym.make('Breakout-v4')
    state = env.reset()
    next_state, reward, terminated, info = env.step(0)
    print("state: " + str(next_state.shape))
    n_bits, pop_size, n_iter, r_cross = 100800, 100, 10, 0.9
    r_mut = 1.0 / float(n_bits)
    best, score = algorithm.genetic_algorithm(breakout_objective, n_bits, pop_size, n_iter, r_cross, r_mut, env)
    run_env(env, best, algorithm.neural_network())


