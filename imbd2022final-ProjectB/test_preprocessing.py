#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import os
import time
import collections

import numpy as np
import pandas as pd
from numpy import insert


# In[ ]:


start_time = time.time()


# In[ ]:


# define the path of the data
folder_path = 'test/'
layers = 25
save_foler = 'test_data'

sg_files = [folder_path+str(i)+'_sg.csv' for i in range(1, layers + 1)]
spike_files = [folder_path+str(i)+'_spike.csv' for i in range(1, layers + 1)]


# In[ ]:


# read the data
print('Reading sg...')
sg_data = [pd.read_csv(file) for file in sg_files]
print('Reading spike...')
spike_data = [pd.read_csv(file) for file in spike_files]

print('\n\n')
print('Seperate each feature to do data preprocessing')
print('-*-*-*' * 18 + '-')

# Define feature index
sg_feature_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
spike_feature_index = ['A', 'B', 'C', 'D']

sg = {}
spike = {}

for i in sg_feature_index:
    s = time.time()
    print(str(i) + '...', end='')
    sg['sg_' + str(i)] = [np.array(sg_data[j][i]) for j in range(layers)]
    print('cost time: ', time.time() - s)
for i in spike_feature_index:
    s = time.time()
    print(str(i) + '...', end='')
    spike['spike_' + str(i)] = [np.array(spike_data[j][i])
                                for j in range(layers)]
    print('cost time: ', time.time() - s)


# In[ ]:


print('\n\n')
print('-*-*-*' * 18 + '-')

sg_feature = list(sg.keys())
spike_feature = list(spike.keys())

print('sg_dict_keys:' + str(sg.keys()))
print('spike_dict_keys:' + str(spike.keys()))


# In[ ]:


def print_data(sg, spike):
    # Show the size of our data
    print('\n\n')
    print('-*-*-*' * 18 + '-')

    for i in sg_feature:
        print(i + '| \t\t', '1:', sg[i][0].size, '\t', int(layers/2)+3, ':', sg[i][int(layers/2)+2].size,
            '\t', int(layers/2)+4, ':', sg[i][int(layers/2)+5].size, '\t', layers, ':', sg[i][layers-1].size,)

    for i in spike_feature:
        print(i + '| \t', '1:', spike[i][0].size, '\t', int(layers/2)+3, ':', spike[i][int(layers/2)+2].size,
            '\t', int(layers/2)+4, ':', spike[i][int(layers/2)+5].size, '\t', layers, ':', spike[i][layers-1].size)


print_data(sg, spike)


# In[ ]:


# If the number of data in one column is smaller than maximum, we'll use the last value of that column
# to fill in to get a same dimension in one feature(column)

# ----sg----
max_num = max([sg[i][j].size for i in sg_feature for j in range(layers)])
print('\n\n')
print('Fill in value to get same dimension in one feature')
print('-*-*-*' * 18 + '-')

for i in sg_feature:
    s = time.time()
    for j in range(len(sg[i])):
        if sg[i][j].size < max_num:
            sg[i][j] = np.append(
                sg[i][j], [[sg[i][j][-1]] * (max_num - sg[i][j].size)])
    print(str(i) + '...', end='')
    print('cost time: ', time.time() - s)


print('--------'*10)
# ----spike----
max_num = max([spike[i][j].size for i in spike_feature for j in range(layers)])
for i in spike_feature:
    s = time.time()
    for j in range(len(spike[i])):
        if spike[i][j].size < max_num:
            spike[i][j] = np.append(
                spike[i][j], [[spike[i][j][-1]] * (max_num - spike[i][j].size)])
    print(str(i) + '...', end='')
    print('cost time: ', time.time() - s)
    
print_data(sg, spike)


# In[ ]:


# Change in sg --> change the data from absolute coordination to relative coordination (distance in each timestep)
print('\n\n')
print('Value change in sg...')
print('-*-*-*' * 18 + '-')
for i in sg_feature:
    s = time.time()
    print(str(i) + '...', end='')
    sg[i] = [np.diff(j, prepend=j[0]) for j in sg[i]]
    print('cost time: ', time.time() - s)


# In[ ]:


# Calculate B's, C's, D's values at A = 0.001, 0.002, 0.003.....
print('\n\n')
print("Calculate B's, C's, D's values at A = 0.001, 0.002, 0.003...")
print('-*-*-*' * 18 + '-')

for i in spike_feature:
    s = time.time()
    print(str(i) + '...', end='')
    temp_copy = copy.deepcopy(spike[i])
    for j in range(len(temp_copy)):
        index = range(3, spike[i][j].size-1, 5)
        value = [(temp_copy[j][k-1] + temp_copy[j][k])/2 for k in index]
        temp_copy[j] = insert(temp_copy[j], index, value)
    spike[i] = temp_copy
    print('cost time: ', time.time() - s)


# In[ ]:


# Change in spike --> change the data from absolute coordination to relative coordination (distance in each timestep)
print('\n\n')
print('Value change in spike...')
print('-*-*-*' * 18 + '-')
for i in spike_feature:
    s = time.time()
    print(str(i) + '...', end='')
    spike[i] = [np.diff(j, prepend=j[0]) for j in spike[i]]
    print('cost time: ', time.time() - s)


# In[ ]:


# Sum up the distance that process during one timestep 0.001
# final_abs_: total path
# fianl_: distance
print('\n\n')
print('Sum up the distance that process during one timestep 0.001')
print('-*-*-*' * 18 + '-')

spike_abs = {}
spike_final = {}

for i in spike_feature:
    s = time.time()
    print(str(i) + '_abs...', end='')
    spike_abs[str(i)] = np.array([np.insert(np.array([sum(abs(spike[i][k][j*3+1:j*3+4]))
                                                      for j in range(0, len(sg['sg_B'][k])-1)]), 0, 0) for k in range(len(spike[i]))])
    print('cost time: ', time.time() - s)

for i in spike_feature:
    s = time.time()
    print(str(i) + '_final...', end='')
    spike_final[str(i)] = np.array([np.insert(np.array([sum((spike[i][k][j*3+1:j*3+4]))
                                                      for j in range(0, len(sg['sg_B'][k])-1)]), 0, 0) for k in range(len(spike[i]))])
    print('cost time: ', time.time() - s)


# In[ ]:


s = time.time()
print('BCD_abs_distance...', end='')
BCD_abs_distance = np.array([[pow(pow(spike_abs['spike_B'][j][i], 2) + pow(spike_abs['spike_C'][j][i], 2)
                                  + pow(spike_abs['spike_D'][j][i], 2), 0.5) for i in range(len(spike_abs['spike_A'][j]))]
                             for j in range(len(spike_abs['spike_A']))])
print('cost time: ', time.time() - s)

s = time.time()
print('BCD_distance...', end='')
BCD_distance = np.array([[pow(pow(spike_final['spike_B'][j][i], 2) + pow(spike_final['spike_C'][j][i], 2)
                              + pow(spike_final['spike_D'][j][i], 2), 0.5) for i in range(len(spike_final['spike_A'][j]))]
                         for j in range(len(spike_final['spike_A']))])
print('cost time: ', time.time() - s)

s = time.time()
print('BDF_distance...', end='')
BDF_distance = np.array([[pow(pow(sg['sg_B'][j][i], 2) + pow(sg['sg_D'][j][i], 2)
                              + pow(sg['sg_F'][j][i], 2), 0.5) for i in range(len(sg['sg_B'][j]))]
                         for j in range(len(sg['sg_B']))])
print('cost time: ', time.time() - s)


# In[ ]:


# names of variables change
for i in spike_feature:
    spike[i] = spike_final[i]


# In[ ]:


print_data(sg, spike)


# In[ ]:


# Creaet lower noise featrue
counter = collections.Counter([])
for i in range(0, len(sg['sg_B'])):
    temp_list = list(abs(np.around(sg['sg_B'][i], 4)))
    counter = counter + collections.Counter(temp_list)
    
print('Frequency of each value in sg_B[0]: ', counter)
upper = counter.most_common()[0][0]
lower = counter.most_common()[1][0]
print('B upper and lower', upper, lower)

work_B = [[1 if abs(sg['sg_B'][i][j]) >= lower-0.0001 and abs(sg['sg_B'][i][j]) <= upper+0.0001 else 0
           for j in range(len(sg['sg_B'][i]))] for i in range(len(sg['sg_B']))]


counter = collections.Counter([])
for i in range(0, len(sg['sg_B'])):
    temp_list = list(abs(np.around(sg['sg_D'][i], 4)))
    counter = counter + collections.Counter(temp_list)
print('Frequency of each value in sg_D[0]: ', counter)
upper = counter.most_common()[0][0]
lower = counter.most_common()[1][0]
print('D upper and lower', upper, lower)

work_D = [[1 if abs(sg['sg_D'][i][j]) >= lower-0.0001 and abs(sg['sg_D'][i][j]) <= upper+0.0001 else 0
           for j in range(len(sg['sg_D'][i]))] for i in range(len(sg['sg_D']))]

spike_abs_B_lower_noise = spike_abs['spike_B'] * work_B
spike_B_lower_noise = spike['spike_B'] * work_B
spike_abs_C_lower_noise = spike_abs['spike_D'] * work_D
spike_C_lower_noise = spike['spike_D'] * work_D

sg_B_lower_noise = np.array(sg['sg_B']) * work_B
sg_D_lower_noise = np.array(sg['sg_D']) * work_D


# In[ ]:


#save data
os.makedirs(save_foler, exist_ok=True)

print('saving sg...')
for i in sg_feature:
    np.save(os.path.join(save_foler, i), np.array(sg[i]))

print('saving spike...')
for i in spike_feature:
    np.save(os.path.join(save_foler, i), np.array(spike[i]))

print('saving spike_abs...')
for i in spike_feature_index:
    name = 'spike_abs_' + i
    np.save(os.path.join(save_foler, name), np.array(spike_abs['spike_' + str(i)]))

print('saving distance...')
np.save(os.path.join(save_foler, 'BCD_abs_distance'), np.array(BCD_abs_distance))
np.save(os.path.join(save_foler, 'BCD_distance'), np.array(BCD_distance))
np.save(os.path.join(save_foler, 'BDF_distance'), np.array(BDF_distance))

print('saving lower noise...')
np.save(os.path.join(save_foler, 'spike_abs_B_lower_noise'), np.array(spike_abs_B_lower_noise))
np.save(os.path.join(save_foler, 'spike_B_lower_noise'), np.array(spike_B_lower_noise))
np.save(os.path.join(save_foler, 'spike_abs_C_lower_noise'), np.array(spike_abs_C_lower_noise))
np.save(os.path.join(save_foler, 'spike_C_lower_noise'), np.array(spike_C_lower_noise))

np.save(os.path.join(save_foler, 'sg_B_lower_noise'), np.array(sg_B_lower_noise))
np.save(os.path.join(save_foler, 'sg_D_lower_noise'), np.array(sg_D_lower_noise))


# In[ ]:


print('saving done!')
print('Total cost: ', time.time() - start_time)

