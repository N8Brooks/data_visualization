# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:09:12 2020

@author: DSU
"""

"""
# ipynb support
%matplotlib qt
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# init to 0s
#x = np.zeros((100, 100, 4), dtype=np.uint8)

# init to random
x = np.random.randint(0, high=255, size=(100, 100, 4), dtype=np.uint8)

# display single image
#plt.imshow(x)

# creating a gif
fig = plt.figure()
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
plt.set_cmap('hot')

# initialization function: plot the background of each frame
ims = []
for i in range(60):
    im = plt.imshow(np.random.randint(0, high=255, size=(100, 100, 4), dtype=np.uint8), aspect='equal')
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True)

plt.show()

#ani.save('static.gif', writer='imagemagick')
ani.save('static.mp4')
