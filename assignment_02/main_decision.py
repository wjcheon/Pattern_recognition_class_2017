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
filename_full_directory = os.path.join(pathname,filename)
img = plt.imread(filename_full_directory)
#%
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
        channel_onevec = {'r':R_channel_onevec, 'g':G_channel_onevec, 'b':B_channel_onevec}
        #
        return channel_onevec
img_channel_seperated = RGB_divider(img)
#target_expectation_r = np.mean(img_channel_seperated['r'])
#target_expectation_g = np.mean(img_channel_seperated['g'])
#target_expectation_b = np.mean(img_channel_seperated['b'])
target_expectation = []
target_expectation.append(np.mean(img_channel_seperated['r']))
target_expectation.append(np.mean(img_channel_seperated['g']))
target_expectation.append(np.mean(img_channel_seperated['b']))

#img = mpimg.imread(filename)
#%% Load Gaussian distrubton of learned 
import pickle
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

loaded_r_dic = load_obj('dic_current_channel_clt_r')
loaded_g_dic = load_obj('dic_current_channel_clt_g')
loaded_b_dic = load_obj('dic_current_channel_clt_b')

#%%
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
                 

answer_sheet = np.zeros((1,len(loaded_r_dic)))
num_channel = 3
#%% R channel 
y = 0 
counter = 0 
for keys_clt_mean, value_clt_mean in loaded_r_dic.items():
    y_ = gaussian(target_expectation[0], value_clt_mean[0], value_clt_mean[1])
    if y < y_:
        y = y_
        answer = keys_clt_mean
        print('activation')
        print(keys_clt_mean)

answer_sheet[0, list(loaded_r_dic.keys()).index(answer)] =answer_sheet[0, list(loaded_r_dic.keys()).index(answer)]  +1
             
#%% B channel 
y = 0 
counter = 0 
for keys_clt_mean, value_clt_mean in loaded_g_dic.items():
    y_ = gaussian(target_expectation[1], value_clt_mean[0], value_clt_mean[1])
    if y < y_:
        y = y_
        answer = keys_clt_mean
        print('activation')
        print(keys_clt_mean)

answer_sheet[0,list(loaded_g_dic.keys()).index(answer)] =answer_sheet[0,list(loaded_g_dic.keys()).index(answer)]  +1
     
#%% R channel 
y = 0 
counter = 0 
for keys_clt_mean, value_clt_mean in loaded_b_dic.items():
    y_ = gaussian(target_expectation[2], value_clt_mean[0], value_clt_mean[1])
    if y < y_:
        y = y_
        answer = keys_clt_mean
        print('activation')
        print(keys_clt_mean)

answer_sheet[0,list(loaded_b_dic.keys()).index(answer)] =answer_sheet[0,list(loaded_b_dic.keys()).index(answer)]  +1

#%%
#print(answer_sheet)
max_index =np.argmax(answer_sheet)
predicted_answer =list(loaded_b_dic.keys())[max_index]
ture_answer = filename
print("Predict : {} // Ture: {}".format(predicted_answer, ture_answer))

#%%





