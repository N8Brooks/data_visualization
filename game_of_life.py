#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 11:43:55 2020

@author: nathan
"""

"""
# ipynb support
%matplotlib qt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import convolve

SIZE = 16
FRAMES = 32

# init to random
#x = np.random.randint(0, high=2, size=(SIZE, SIZE), dtype=np.uint8)

# init to glider
x = np.zeros((SIZE,SIZE), dtype=np.uint8)
x[1,3] = 1
x[2,3] = 1
x[2,1] = 1
x[3,3] = 1
x[3,2] = 1

# neighbor sum matrix
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

# setting up plt
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('hot')

# survival function
f = np.vectorize(lambda a, b: np.uint8(b == 3 or (a == 1 and b == 2)))

# running through frames
ims = [[plt.imshow(x, aspect='equal')]]
for i in range(1, FRAMES):
    x = f(x, convolve(x, kernel, mode='constant'))
    ims.append([plt.imshow(x, aspect='equal')])

# animation
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
plt.show()

# saving
#ani.save('game_of_life.gif', writer='imagemagick')
ani.save('game_of_lifex.mp4')
