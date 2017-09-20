# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:46:07 2017

@author: WJCHEON
"""
#%%
import numpy as np 
import matplotlib.pyplot as plt
number_of_rand = 2000
#random_variable = np.round(np.random.rand(number_of_rand)*20)
random_variable = np.floor(np.random.rand(number_of_rand)*20)
#
plt.figure()
plt.hist(random_variable,20)
plt.show()
#%%
number_of_sample = 10
number_of_conduct = 500
#smaple_index = []

sample_mean = []
for iter2 in range(number_of_conduct):
    samples = np.zeros(number_of_sample)
    for iter1 in range(number_of_sample):
        sample_index_buffer = np.random.randint(number_of_rand)
        sample_buffer = random_variable[sample_index_buffer]
        #smaple_index.append(sample_index_buffer)
        #samples.append(sample_buffer)
        samples[iter1] = sample_buffer
        sample_mean_buffer = np.mean(samples)
        #print(samples)
    if iter2%10 == 0:
        print("{} of {} is conducted".format(iter2, number_of_conduct))
    del samples
    sample_mean.append(sample_mean_buffer)
#%%

plt.figure()
plt.hist(sample_mean,20)
plt.show()