# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:43:59 2017

@author: Wonjoong Cheon
"""
#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os 
#%% Set parameters
number_of_channel = 3
result_folder_name = 'result_modified'
#%%
from tkinter import filedialog
from tkinter import *
 
Tk_gui_instance = Tk();
Tk_gui_instance.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print(Tk_gui_instance.filename)
pathname, filename = os.path.split(Tk_gui_instance.filename)
#
result_folder_directory = os.path.join(pathname,result_folder_name)
os.makedirs(result_folder_directory, exist_ok=True)
#
img = plt.imread(filename)
#img = mpimg.imread(filename)
#plt.imshow(img)
# Seperate channel each variable
img_channel_R = np.ndarray.astype(np.array(img[:,:,0]), np.float32)
img_channel_G = np.ndarray.astype(np.array(img[:,:,1]), np.float32)
img_channel_B = np.ndarray.astype(np.array(img[:,:,2]), np.float32)

#%%
for buffer_img, iter_val in zip([img_channel_R, img_channel_G, img_channel_B], ['r', 'g', 'b']):
    #target_img = img_channel_G.copy()
    target_img = buffer_img.copy()
    target_img_one_vec = target_img[-1]
    #
    #%% Histogram 
    #histogram parameter set
    nbins = 50
    n, bins, patches = plt.hist(target_img_one_vec, bins=nbins, facecolor='g', alpha=0.75)
    target_img_one_vec_mean = np.mean(target_img_one_vec)
    target_img_one_vec_var = np.var(target_img_one_vec, ddof=0)
    target_img_one_vec_std = np.std(target_img_one_vec, ddof=0)
    
    plt.xlabel('Pixel intensity')
    plt.ylabel('Number of count')
    plt.title('Histogram\nMean: {0:.4f}, Variance: {1:.4f}'.format(target_img_one_vec_mean,target_img_one_vec_var ))
    plt.grid(True)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([0, 255, 0, 1000])
    #
    #np.sum(n)
    #len(target_img_one_vec)
    #
    filename_save_hist = 'histogram_of_'+iter_val+'_'+filename
    directory_save_hist = os.path.join(result_folder_directory,filename_save_hist)
    plt.savefig(directory_save_hist)
    #
    plt.show()
    #%% (n/number of pixel) histogram 
    weights = np.ones_like(target_img_one_vec)/float(len(target_img_one_vec))
    n_norm, bins_norm, patches_norm = plt.hist(target_img_one_vec, bins=nbins, weights=weights, facecolor='g', alpha=0.75)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Probability')
    plt.title('Normalized histogram\nMean: {0:.4f}, Variance: {1:.4f}'.format(target_img_one_vec_mean,target_img_one_vec_var))
    plt.grid(True)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([0, 255, 0, 1000])
    #plt.show()
    filename_save_hist_norm = 'Normalized_histogram_of_'+iter_val+'_'+filename
    directory_save_hist_norm = os.path.join(result_folder_directory,filename_save_hist_norm)
    plt.savefig(directory_save_hist_norm)
    #
    plt.show()
    np.sum(n/len(target_img_one_vec))
    #
    np.sum(n_norm)
    #%% Find minimum and maximum value 
    target_img_one_vec_mean = np.mean(target_img_one_vec)
    target_img_one_vec_std = np.std(target_img_one_vec)
    print("Mean: {0:.4f} \n  Variance: {1:.4f} \n".format(target_img_one_vec_mean, target_img_one_vec_var))
    #%%
    weights = np.ones_like(target_img_one_vec)/float(len(target_img_one_vec))
    n_norm, bins_norm, patches_norm = plt.hist(target_img_one_vec, bins=nbins, weights=weights, facecolor='g', alpha=0.75)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Probability')
    plt.title('Normalized histogram')
    plt.grid(True)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([0, 255, 0, 1000])
    np.sum(n_norm)
    #
    #gaussian_val = gaussian_val*np.max(n)
    import matplotlib.mlab as mlab
    import math
    mu = target_img_one_vec_mean
    sigma = target_img_one_vec_std
    x = np.linspace(0,300, num=300)
    y = mlab.normpdf(x, mu, sigma)
    np.sum(y)
    plt.plot(x,mlab.normpdf(x, mu, sigma))
    #plt.show()
    #
    filename_save_hist_norm_add_gaussian = 'Normalized_histogram_with_gaussian_'+iter_val+'_'+filename
    directory_save_hist_norm_add_gaussian = os.path.join(result_folder_directory,filename_save_hist_norm_add_gaussian)
    plt.savefig(directory_save_hist_norm_add_gaussian)
    #
    plt.show()
#%%
#s = np.random.normal(target_img_one_vec_mean, target_img_one_vec_std, 1000)
#count, bins, ignored = plt.hist(s, 30, normed=True)
#width= bins[1]-bins[0]
#np.sum(count*width)
#%%
#def gaussian(x, mu, sig):
#        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
#    #%%
#gaussian_val = gaussian(np.linspace(0, 1000,1000), target_img_one_vec_mean, target_img_one_vec_std)
#np.sum(gaussian_val )