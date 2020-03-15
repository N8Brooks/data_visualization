# -*- coding: utf-8 -*-
"""
Simulation of covid-19 based on one week infection period for each individual
"""

from random import choices, sample
from array import array
import pandas as pd
from tqdm import trange

# american population / initially infected individuals / r0
# anecdote: herd immunity = (1 - 1 / r0)
count = 330000000
sick = 1000
r0 = 1.5

# dataframe for recording amount of new infections
df = pd.DataFrame([{'infected':sick}])

'''
Array of all american individuals, defaults to not immune (0)
0 = immune
1 = infectable
'''
population = array('B', [True] * count)

# initial infected people
infected = set(sample(range(count), sick))
for x in infected:
    population[x] = False

# model infection rate until no more people are infected
i = 1
while len(infected):
    print(len(infected))
    
    # every infected individual can infect r0 healthy people their first week
    infected = set(x for x in choices(range(count),k=int(len(infected) * r0)) if population[x])
    
    # set newly infected individuals to infected
    for x in infected:
        population[x] = False
    
    # record how many people were infected this week
    df.loc[i] = (len(infected),)
    i += 1

# amount of total people infected for an entire year
df['cumulative'] = df['infected'].cumsum()
