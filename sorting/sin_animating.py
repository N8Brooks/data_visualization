#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:36:21 2020

@author: nathan
"""

from math import pi, sin

def sin_array(size):
    if size == 1 or size == 2: return [1]
    wh = (size - 1) / 2
    ret = [round(wh * sin(pi * x / wh) ) for x in range(size)]
    m = min(ret, default=1) - 1
    return [x - m for x in ret]

from sorting import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation

order = [(slow_sort, 128, 16,),
         (stooge_sort, 256, 16,),]
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
    frames = [([], 'Increasing_size')]
    
    for algo, target_n, frame_ratio in order:
        # bump data up to size
        tmp = list()
        for i in range(len(data), target_n + 1):
            data = sin_array(i)
            tmp.append(data[:])
        # add ~4 or less secs of increasing size
        if 0 < len(tmp) < 240:
            frames.extend((x, 'Increasing_size') for sub in \
                          zip(*[tmp]*(180 // len(tmp))) for x in sub)
        elif tmp:
            frames.extend((x, 'Increasing_size') for x in \
                          tmp[::max(1, len(tmp)//240)] + [tmp[-1]])

        # sort data with algo
        # x seconds sorting
        tmp = [data[:]]
        for update in algo(data):
            if update != tmp[-1]:
                tmp.append(update[:])
        
        print(f'{algo.__name__}\t\t{len(tmp)}')
        # slice frames and pad with 60 frames of it sorted
        frames.extend((x, algo.__name__.capitalize()) for x in \
                      tmp[::frame_ratio] + 60*[sorted(tmp[-1])])
    
        # returning to size
        tmp = list()
        if algo.__name__ != 'msd_sort':
            target = sin_array(target_n)
            while target != data:
                for i in range(len(data)):
                    if data[i] > target[i]:
                        data[i] -= 1
                    elif data[i] < target[i]:
                        data[i] += 1
                    tmp.append(data[:])
        step = max(1, len(tmp) // 240)
        frames.extend((x, 'Resetting',) for x in tmp[::step] + [tmp[-1]])
    
    
    fig = plt.figure(figsize=(19.2, 10.8))
    plt.axis('off')
    bars = plt.bar([], [], width=1., color='black')
    plt.tight_layout(pad=0)
    pbar = tqdm(total=len(frames))
    
    def animate(i):
        global bars
        pbar.update()
        data = frames[i][0]
        for x, y in zip(bars, data):
            x.set_height(y)
        for i in range(len(data), len(bars)):
            bars[i].set_color('black')
        for j in range(len(bars), len(data)):
            bars += plt.bar([len(bars)], [data[j]], width=1., color='black')
        plt.title(frames[i][1], loc='left', fontdict={'color':'black'})
        
    
    anim=animation.FuncAnimation(fig, animate, blit=False, \
                                 frames=len(frames), interval=1)
    
    anim.save(f'sin_animation.mp4',\
              writer=animation.FFMpegWriter(fps=60))
