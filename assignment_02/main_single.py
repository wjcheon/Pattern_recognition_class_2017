# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:26:24 2017

@author: WJCHEON
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
filename_full_directory = os.path.join(pathname,filename)
img = plt.imread(filename_full_directory)
#img = mpimg.imread(filename)
#plt.imshow(img)
# Seperate channel each variable
img_channel_R = np.ndarray.astype(np.array(img[:,:,0]), np.float32)
img_channel_G = np.ndarray.astype(np.array(img[:,:,1]), np.float32)
img_channel_B = np.ndarray.astype(np.array(img[:,:,2]), np.float32)
#%%
def clt_wjcheon(random_variables_, num_conduct_, num_sampels_):
    import numpy as np 
    np.asarray(random_variables_)
    random_variable = random_variables_.flat
    number_of_sample = num_sampels_
    number_of_conduct = num_conduct_
    number_of_index = len(random_variable)
    #smaple_index = []

    sample_mean = []
    for iter2 in range(number_of_conduct):
        samples = np.zeros(number_of_sample)
        for iter1 in range(number_of_sample):
            sample_index_buffer = np.random.randint(number_of_index)
            sample_buffer = random_variable[sample_index_buffer]
            #smaple_index.append(sample_index_buffer)
            #samples.append(sample_buffer)
            #print(sample_buffer)
            samples[iter1] = sample_buffer
            sample_mean_buffer = np.mean(samples)
                   #print(samples)
            if iter2%10 == 0:
                print("{} of {} is conducted".format(iter2, number_of_conduct))
        del samples
        sample_mean.append(sample_mean_buffer)
    return sample_mean
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
import glob
import os
img_list = glob.glob(os.path.join(pathname,'*jpg'))
number_of_files = len(img_list)
dic_current_channel = {}
dic_current_channel_clt = {}
for iter1 in range(number_of_files):
    imgs_buffer = plt.imread(img_list[iter1])
    _, imgs_filename = os.path.split(img_list[iter1])
    R_channel_onevec, G_channel_onevec, B_channel_onevec = RGB_divider(imgs_buffer)
    print(imgs_filename)
    dic_current_channel[imgs_filename] = R_channel_onevec
    #
    num_conduct = 5000
    num_samples = 100 
    r_clt = clt_wjcheon(R_channel_onevec,num_conduct,num_samples)
    dic_current_channel_clt[imgs_filename+'_clt'] = r_clt
    del r_clt
    
#%%
#num_clt_data = len(dic_current_channel_clt)
nbins = 200 
counter = 0 
c_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
plt.figure()
for keys, values in dic_current_channel_clt.items(): 
    n, bins, patches = plt.hist(dic_current_channel_clt[keys], bins=nbins, facecolor=c_map[counter], alpha=0.75)
    counter = counter + 1
plt.show()
    #%%
dic_current_channel_clt_mean_std = {}
for keys, values in dic_current_channel_clt.items(): 
    clt_data_buffer = dic_current_channel_clt[keys]
    mean_buffer = np.mean(clt_data_buffer)
    std_buffer = np.std(clt_data_buffer)
    dic_current_channel_clt_mean_std[keys] = [mean_buffer, std_buffer]
    
    
    #%%
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
                 
target_expectation = 235
y =0
answer = 'None'
for keys_clt_mean, value_clt_mean in dic_current_channel_clt_mean_std.items():
    y_ = gaussian(target_expectation, value_clt_mean[0], value_clt_mean[1])
    if y < y_:
        y = y_
        answer = keys_clt_mean
        print('activation')
        print(keys_clt_mean)


