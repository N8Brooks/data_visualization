#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:10:41 2020

@author: nathan
"""

"""
# ipynb support
%matplotlib qt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# init to random
x = np.random.randint(0, high=255, size=(100, 100, 4), dtype=np.uint8)

# creating a gif
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('hot')

# initialization function: plot the background of each frame
ims = [[plt.imshow(x, aspect='equal')]]
for i in range(59):
    y = np.zeros((100,100,4), dtype=np.uint8)
    y[:-1,:-1] += x[1:,1:] + x[1:,:-1] + x[:-1,1:]
    y[1:,1:] += x[:-1,:-1] + x[1:,:-1] + x[:-1,1:]
    y[1:,:-1] += x[:-1,1:]
    y[:-1,1:] += x[1:,:-1]
    y //= 8
    im = plt.imshow(y, aspect='equal')
    x = y
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

plt.show()

#ani.save('neighbor_average.gif', writer='imagemagick')
ani.save('neighbor_average.mp4')
