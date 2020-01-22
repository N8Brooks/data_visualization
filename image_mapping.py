# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:09:12 2020

@author: DSU
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# init to 0s
#x = np.zeros((100, 100, 4), dtype=np.uint8)

# init to random
x = np.random.randint(0, high=255, size=(100, 100, 4), dtype=np.uint8)

# display single image
plt.imshow(x)


# creating a gif
fig = plt.gcf()

# Show first image - which is the initial board
im = plt.imshow(x)

def animate(frame):
    im.set_data(np.random.randint(0, high=255, size=(100, 100, 4), dtype=np.uint8))
    return im,

anim = animation.FuncAnimation(fig, animate, frames=200, 
                               interval=50)

plt.show()

anim.save('thingy.mp4')
