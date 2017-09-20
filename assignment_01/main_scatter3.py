# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:24:33 2017

@author: WJCHEON
"""

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
#
#filename = '5000won.jpg'
from tkinter import filedialog
from tkinter import *
 
Tk_gui_instance = Tk();
Tk_gui_instance.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
img = mpimg.imread(Tk_gui_instance.filename)
[pathname, filename] = os.path.split(Tk_gui_instance.filename)
plt.imshow(img)
# Seperate channel each variable
img_channel_R = np.ndarray.astype(np.array(img[:,:,0]), np.float32)
img_channel_G = np.ndarray.astype(np.array(img[:,:,1]), np.float32)
img_channel_B = np.ndarray.astype(np.array(img[:,:,2]), np.float32)
#
img_channel_R_onevec = img_channel_R[-1]
img_channel_G_onevec = img_channel_G[-1]
img_channel_B_onevec = img_channel_B[-1]
#
#%%
# 3d scatter : https://matplotlib.org/examples/mplot3d/scatter3d_demo.html
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(img_channel_R_onevec, img_channel_G_onevec, img_channel_B_onevec)
angle = 190
ax.view_init(30,angle)
#
plt.xlabel('R channel')
plt.ylabel('G channel')
filename_save = os.path.join(pathname,'3dscatter'+'_'+filename)
plt.savefig(filename_save)