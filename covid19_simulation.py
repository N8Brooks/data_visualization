# -*- coding: utf-8 -*-
"""
Simulation of spread of covid-19 based on two week infection period
"""

from random import choices, sample
from array import array
import pandas as pd
from tqdm import trange

# american population / initially infected individuals
count = 330000000
sick = 1000
r0 = 1.5

# dataframe for recording amount of new infections
df = pd.DataFrame([{'infected':sick}])

'''
Array of all american individuals, defaults to never sick (0)
0 = never sick
1 = currently sick
2 = previously sick
'''
population = array('b', [0] * count)

# correction factor for infection based on r0 and 2-week infection period
inf_rate = 2 / r0

# initial sick people
one_week, two_week = set(sample(range(count), sick)), set()
for x in one_week:
    population[x] = 1

# model infection rate until no more people are infected
i = 1
while (len(one_week) + len(two_week)):
    print(len(one_week) + len(two_week))
    # individuals who have healed and are now immune
    # even though they may not be immune ... this is just a model afterall
    for x in two_week:
        population[x] = 2
    
    # every infected individual can infect one healthy person a week
    # based on r0 = 2.2 and infectivity lasts for 2 weeks. 2/2=1
    one_week, two_week=set(x for x in choices(range(count),k=int((len(one_week)
        + len(two_week)) // inf_rate)) if population[x] == 0), one_week
    
    # set newly infected individuals to infected
    for x in one_week:
        population[x] = 1
    
    # record how many people were infected this week
    df.loc[i] = (len(one_week),)
    i += 1

# amount of total people infected for an entire year
df['cumulative'] = df['infected'].cumsum()
