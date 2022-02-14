import numpy as np

class Evolutionary:

    #CITE: this algorithm has been implemented with help of the tutorial available here:
    # https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    # I am solely using this as a starting point and the algorithm will be modified in the future
    def genetic_algorithm(self, objective, n_bits, pop_size, n_iter, r_cross, r_mut):
        # create population of size pop_size, where each item is a bitstring of size n_bits
        population = [np.random.randint(0, 2, n_bits).tolist() for _ in range(pop_size)]
        best, best_eval = 0, objective(population[0])
        for generation in range(n_iter):
            scores = [objective(c) for c in population]
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
                bitstring[i] = 1 - bitstring[i]

if __name__ == '__main__':
    algorithm = Evolutionary()
    algorithm.selection([1, 2], [3, 4])
