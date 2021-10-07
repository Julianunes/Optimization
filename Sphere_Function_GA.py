import numpy as np
import matplotlib.pyplot as plt

def generate_population(size, dim, inf_boundary, sup_boundary):
    population = []
    for i in range(size):
        population.append(inf_boundary + (sup_boundary - inf_boundary)*np.random.rand(dim))

    return population

def apply_function(individual):
    return individual[0]**2 + individual[1]**2

def choice_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum

    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness;
        normalized_fitness_sum += offset*len(sorted_population)

    draw = np.random.rand()

    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness/normalized_fitness_sum
        probability = 1 - probability
        accumulated += probability

        if draw < accumulated:
            return individual


def sort_population_by_fitness(population):
    return sorted(population, key=apply_function)

def crossover(individual_a, individual_b):
    n = len(individual_a)
    individual_c = np.zeros(n)

    for i in range(n):
        individual_c[i] = (individual_a[i] + individual_b[i])/2

    return individual_c

def mutation(individual, inf_boundary, sup_boundary):

    n = len(individual)

    for i in range(n):
        draw = np.random.rand()
        if draw > 0.5:
            individual[i] += -0.05 + 0.1*np.random.rand()
            individual[i] = min(max(individual[i], inf_boundary), sup_boundary)

    return individual

def make_next_generation(previous_population, inf_boundary, sup_boundary):

    next_generation = []
    sorted_population = sort_population_by_fitness(previous_population)
    population_size = len(sorted_population)
    fitness_sum = 0
    
    for i in range(population_size):
        fitness_sum += apply_function(sorted_population[i])

    for i in range(population_size):
        first_individual = choice_by_roulette(sorted_population, fitness_sum)
        second_individual = choice_by_roulette(sorted_population, fitness_sum)

        new_individual = crossover(first_individual, second_individual)
        next_generation.append(mutation(new_individual, inf_boundary, sup_boundary))

    return next_generation

fitness_state_list = []

for runs in range(30):
    generations = 100
    size = 20
    dim = 2
    linf = -2
    lsup = 2

    population = generate_population(size, dim, linf, lsup)

    for generation in range(generations):
        
        population = make_next_generation(population, linf, lsup)

    best_individual = sort_population_by_fitness(population)[-1]
    fitness_state_list.append(apply_function(best_individual))

plt.boxplot(fitness_state_list)
plt.show()

