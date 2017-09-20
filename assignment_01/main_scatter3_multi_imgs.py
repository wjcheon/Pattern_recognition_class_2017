# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 17:52:35 2017

@author: WJCHEON
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#%%
def RGB_divider(img_):
    #
    import numpy as np
    #
    if len(np.shape(img_)) == 3:
        R_channel = img_[:,:,0]
        G_channel = img_[:,:,1]
        B_channel = img_[:,:,2]
        #
        R_channel_onevec = R_channel[-1]
        G_channel_onevec = G_channel[-1]
        B_channel_onevec = B_channel[-1]
        return R_channel_onevec, G_channel_onevec, B_channel_onevec

#%%
img_1000 = mpimg.imread('./1000won.jpg')
img_5000 = mpimg.imread('./5000won.jpg')
img_10000 = mpimg.imread('./10000won.jpg')
img_50000 = mpimg.imread('./50000won.jpg')
#
img_1000_R,img_1000_G,img_1000_B = RGB_divider(img_1000)
img_5000_R,img_5000_G,img_5000_B = RGB_divider(img_5000)
img_10000_R,img_10000_G,img_10000_B = RGB_divider(img_10000)
img_50000_R,img_50000_G,img_50000_B = RGB_divider(img_50000)
#%% 
# 3d scatter : https://matplotlib.org/examples/mplot3d/scatter3d_demo.html
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#    
ax.scatter(img_1000_R, img_1000_G, img_1000_B, c='r', marker='o')
ax.scatter(img_5000_R, img_5000_G, img_5000_B, c='g', marker='^')
ax.scatter(img_10000_R, img_10000_G, img_10000_B, c='b', marker='o')
ax.scatter(img_50000_R, img_50000_G, img_50000_B, c='k', marker='^')
#
angle = 100
ax.view_init(30,angle)
#
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#
plt.show()
