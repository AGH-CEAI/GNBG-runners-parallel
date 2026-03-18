import numpy as np

class HybridAdaptiveDifferentialEvolutionEnhanced:
    def __init__(self, dim=10):
        self.dim = dim
        self.population_size = 3 * dim + 5
        self.bounds = (-100.0, 100.0)
        self.F = np.random.uniform(0.4, 0.9, size=self.population_size)
        self.CR = np.random.uniform(0.2, 0.7, size=self.population_size)
        self.historical_best = []
        self.indices = [np.delete(np.arange(self.population_size), i) for i in range(self.population_size)]
        self.dynamic_scaling = np.ones(self.population_size) * 0.1

    def initialize_population(self):
        return np.random.uniform(*self.bounds, size=(self.population_size, self.dim))

    def calculate_fitness(self, func, population):
        return [func(ind) for ind in population]

    def mutation(self, population, i, F):
        (r1, r2, r3) = np.random.choice(self.indices[i], 3, replace=False)
        best_index = np.argmin([f[0] for f in self.historical_best]) if self.historical_best else None
        mutant = population[r1] + F * (population[r2] - population[r3])
        if best_index is not None:
            adaptive_factor = np.exp(-self.dynamic_scaling[i] * len(self.historical_best))
            mutant += adaptive_factor * (self.historical_best[best_index][1] - population[r1])
        return np.clip(mutant, *self.bounds)

    def crossover(self, target, mutant, CR):
        trial = np.copy(target)
        j_rand = np.random.randint(0, self.dim)
        for j in range(self.dim):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def selection(self, func, population, fitness, i, trial):
        trial_fitness = func(trial)
        if trial_fitness < fitness[i]:
            (population[i], fitness[i]) = (trial, trial_fitness)
            if trial_fitness < self.f_opt:
                (self.f_opt, self.x_opt) = (trial_fitness, trial)
        return (population, fitness)

    def update_historical_best(self):
        if len(self.historical_best) == 0 or self.f_opt < min([h[0] for h in self.historical_best]):
            self.historical_best.append((self.f_opt, np.copy(self.x_opt)))
        self.historical_best = self.historical_best[-10:]

    def update_parameters(self):
        for i in range(self.population_size):
            self.F[i] = min(1.0, max(0.4, self.F[i] + np.random.normal(0, 0.1)))
            self.CR[i] = min(1.0, max(0.2, self.CR[i] + np.random.normal(0, 0.1)))
            if np.random.rand() > 0.95:
                self.dynamic_scaling[i] *= np.exp(np.random.uniform(-1, 1))

    def __call__(self, func, stopping_condition):
        population = self.initialize_population()
        fitness = self.calculate_fitness(func, population)
        (self.f_opt, self.x_opt) = self._initial_best_selection(fitness, population)
        while not stopping_condition():
            (population, fitness) = self._evolve_population(func, population, fitness)
            self.update_historical_best()
            self.update_parameters()
        return (self.f_opt, self.x_opt)

    def _initial_best_selection(self, fitness, population):
        best_index = np.argmin(fitness)
        return (min(fitness), population[best_index])

    def _evolve_population(self, func, population, fitness):
        for i in range(self.population_size):
            mutant = self.mutation(population, i, self.F[i])
            trial = self.crossover(population[i], mutant, self.CR[i])
            (population, fitness) = self.selection(func, population, fitness, i, trial)
        return (population, fitness)