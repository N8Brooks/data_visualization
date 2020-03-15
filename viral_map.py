# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:07:08 2020

@author: DSU
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from random import gauss, randrange, random

count = int(330e6)
r = 2.3

csqrt = round(sqrt(count))
pop = np.ones((csqrt, csqrt,), dtype=np.bool)
infected = set()
intr, extr = int(r), r % 1

while len(infected) < 1:
    infection = (randrange(csqrt), randrange(csqrt),)
    infected.add(infection)
    pop[infection[1], infection[0]] = False

df = pd.DataFrame(columns=['infected'])

while infected:
    print(len(infected))
    #plt.imshow(pop)
    df.loc[len(df)] = (len(infected),)
    tmp = set()
    
    for y, x in infected:
        # people come in contact with ~20 people a day
        # people travel ~once a month (might need to correct more for travel)
        for _ in range(intr + (random() <= extr)):
            ny = round(gauss(y, 2)) % csqrt
            nx = round(gauss(x, 2)) % csqrt
            if x == nx and y == ny:
                nx = randrange(csqrt)
                ny = randrange(csqrt)
            if pop[ny, nx]:
                pop[ny, nx] = False
                tmp.add((nx, ny,))
    
    infected = tmp
    