#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:31:05 2020

@author: nathan
"""

from sorting import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

order = [(bogo_sort, 8, 64,),
         (perm_sort, 8, 32,),
         (slow_sort, 256, 16,),
         (stooge_sort, 256, 16,),
         (bubble_sort, 512, 64,),
         (insertion_sort, 512, 64,),
         (cocktail_sort, 512, 64,),
         (odd_even_sort, 512, 64,),
         (comb_sort, 1024, 4,),
         (shell_sort, 1024, 16,),
         (pancake_sort, 1024, 2,),
         (binsertion_sort, 1024, 1,),
         (selection_sort, 1024, 1,),
         (spaghetti_sort, 1024, 1,),
         (cycle_sort, 1024, 1,),
         (strand_sort, 1024, 2,),
         (circle_sort, 1024, 8,),
         (bitonic_sort, 1024, 16,),
         (heap_sort, 1720, 16,),
         (quick_sort, 1720, 8,),
         (merge_sort, 1720, 8,),
         (tim_sort, 1720, 4,),
         (bucket_sort, 1720, 4,),
         (counting_sort, 1720, 2,),
         (lsd_sort, 1720, 8,),
         (msd_sort, 1720, 8,),]

if __name__ == '__main__':
    data = list()
    frames = [([], f'Increasing_size -> {order[0][1]}')]
    
    for algo, target_n, frame_ratio in order:
        # bump data up to size
        data.sort()
        tmp = list()
        while len(data) < target_n:
            data.append(len(data) + 1)
            tmp.append(data[:])
        # add ~4 or less secs of increasing size
        if 0 < len(tmp) < 240:
            frames.extend((x, f'Increasing_size -> {target_n}') for sub in \
                          zip(*[tmp]*(180 // len(tmp))) for x in sub)
        elif tmp:
            frames.extend((x, f'Increasing_size -> {target_n}') for x in \
                          tmp[::max(1, len(tmp)//240)] + [tmp[-1]])
        
        # add ~4 or less secs of suffling
        every = max(1, len(data) // 240)
        for i, data in enumerate(fisher_yates(data)):
            if i % every == 0:
                frames.append((data[:], 'Shuffling'))

        # sort data with algo
        # x seconds sorting
        tmp = [data[:]]
        for update in algo(data):
            if update != tmp[-1]:
                tmp.append(update[:])

        print(f'{algo.__name__}\t\t{len(tmp)/frame_ratio/60} seconds')
        # slice frames and pad with 60 frames of it sorted
        frames.extend((x, algo.__name__.capitalize()) for x in \
                      tmp[::frame_ratio] + 60*[sorted(tmp[-1])])
    
    fig = plt.figure(figsize=(19.2, 10.8))
    plt.axis('off')
    bars = plt.bar([], [], width=1., color='black')
    
    pbar = tqdm(total=len(frames))
    
    def animate(i):
        global bars
        pbar.update()
        data = frames[i][0]
        for x, y in zip(bars, data):
            x.set_height(y)
        for j in range(len(bars), len(data)):
            bars += plt.bar([len(bars)], [data[j]], width=1., color='black')
        plt.tight_layout(pad=0)
        plt.title(frames[i][1], loc='left')
        
    
    anim=animation.FuncAnimation(fig, animate, blit=False, \
                                 frames=len(frames), interval=1)
    
    anim.save(f'final_animation3.mp4',\
              writer=animation.FFMpegWriter(fps=60))
