# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:23:31 2020

@author: DSU
"""

from random import choices
import pandas as pd
from tqdm import trange
import multiprocessing as mp

def virus_perm_immunity(r0, sick=2600, population=330000000):
    df = pd.DataFrame([{'infected':sick}])
    healthy = population - sick
    
    #pbar = trange(1, 260)
    for t in range(1,260):
        tmp = healthy
        for x in choices(range(population), k=round(r0 * sick)):
            if x < healthy:
                healthy -= 1
        df.loc[t] = sick = tmp - healthy
        #pbar.desc=str(sick)
    
    return list(df['infected'])

def virus_temp_immunity(r0, sick=100, population=1000, vacc=0.5):
    df = pd.DataFrame([{'infected':sick}])
    # vaccinated individuals are immune
    healthy = population - round(population * vacc) - sick
    
    pbar = trange(1, 260)
    for t in pbar:
        tmp = healthy
        for x in choices(range(population), k=round(r0 * sick)):
            if x < healthy:
                healthy -= 1
        df.loc[t] = sick = tmp - healthy
        pbar.desc=str(sick)
        
        # people who loose immunity
        if len(df) > 50:
            healthy += int(df.loc[t - 50]['infected'])
    
    return df

# amount of total people infected for an entire year
#df = virus_perm_immunity(1.5, sick=1000, population=330000000)
#df = virus_temp_immunity(2.3, sick=2000, population=66440000, vacc=0.5)
#df.plot()
#df['c'] = df['infected'].cumsum()
#df['ratio'] = df['cumulative'].pct_change().add(1)


if __name__ == '__main__':
    import numpy as np
    
    with mp.Pool(24) as p:
        results = [p.apply_async(virus_perm_immunity, args=(float(i),)) for i in np.arange(0.1, 2.41, 0.1)]
        p.close()
        p.join()
    
    df = pd.DataFrame()
    for i, col in zip(np.arange(0.1, 2.41, 0.1), results):
        df['{:.2}'.format(i)] = col.get()
    
    df.to_csv('variable_r.csv')














