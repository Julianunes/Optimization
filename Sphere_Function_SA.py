import numpy as np
import matplotlib.pyplot as plt

def s_annealing(random_start, cost_function, random_neighbor, acceptance, temperature, maxsteps = 10000):

    min_int = -2
    max_int = 2
    dim = 2

    states = []
    costs = []
    
    state = random_start(min_int, max_int, dim)
    cost = cost_function(state)
    states.append(state)
    costs.append(cost)

    for i in range(maxsteps):
        fraction = i/float(maxsteps)
        T = temperature(fraction)

        new_state = random_neighbor(state, fraction, min_int, max_int, dim)
        new_cost = cost_function(new_state)

        if acceptance(cost, new_cost, T) > np.random.rand():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
    
    return state, cost, states, costs;

def random_start(mini, maxi, dim):
    a, b = mini, maxi
    s = []
    s = a + (b - a)*np.random.rand(dim)
    return s

def cost_function(s):
    return s[0]**2 + s[1]**2

def temperature(f):
    return max(0.01, min(1, 1-f))

def random_neighbor(s, fraction, mini, maxi, dim):
    amplitude = (maxi - mini) * fraction / 10
    delta = []
    delta = (-amplitude/2.) + amplitude * np.random.rand(dim)
    return clip(s + delta, mini, maxi)

def clip(s, mini, maxi):
    return max(mini, min(s[0], maxi)), max(mini, min(s[1], maxi))  

def acceptance(c, new_c, t):
    if new_c < c:
        return 1
    else:
        return np.exp(-(new_c)/t)

state_f_list= []
c_f_list = []

for runs in range(30):
    state_f, c_f, states_f, costs_f = s_annealing(random_start, cost_function, random_neighbor, acceptance, temperature, maxsteps = 10000)
    state_f_list.append(state_f_list)
    c_f_list.append(c_f)


plt.boxplot(c_f_list)
plt.show()




