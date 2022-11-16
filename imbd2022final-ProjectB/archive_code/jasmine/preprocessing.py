# %%
#Import Module
import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import time
import copy


# %%
# Get files name
path1 = 'train1/'
lenght = 47

sg_files = []
spike_files = []

for i in range(1, lenght):
    sg_files.append( path1 + str(i) + '_sg.csv')
    spike_files.append( path1 + str(i) + '_spike.csv')
    
print(sg_files)
print(spike_files)


# %%
# Read file and seperate each feature to do data preprocessing
s = time.time()
print('Concating sg...')
sg_data = [pd.read_csv(i) for i in sg_files]

print('Concating spike...')
spike_data = [pd.read_csv(i) for i in spike_files]

print('Reading MaxWear...')
output = pd.read_csv(path1+'00_Wear_data.csv')
output.drop(['Index'], axis="columns", inplace=True)

print('sg_data: ' + str(len(sg_data)))
print('spike_data: ' + str(len(spike_data)))

print('cost time:', time.time() - s)


# %%
# Define feature index
Sg_Feature_Index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
Spike_Feature_Index = ['A', 'B', 'C', 'D']

s = time.time()

# Concat data
sg_data_concat = pd.concat(sg_data, axis=1)
spike_data_concat = pd.concat(spike_data, axis=1)

print('sg_data_concat' + str(sg_data_concat.shape))
print('spike_data_concat' + str(spike_data_concat.shape))

# Create dict then use the last value of that column to fill in to get a same dimention in one featrue(column)
sg = {}
spike = {}

for i in Sg_Feature_Index:
  print('fillna sg_' + str(i) + '...')
  data = sg_data_concat[i].copy()
  for j in range(0, lenght-1):
      initial_data = data.iloc[:, j]
      if True in initial_data.isnull().values:
          last_value = initial_data[np.where(
              initial_data.isnull().values == True)[0][0]-1]
          data.iloc[:, j] = sg_data_concat[i].iloc[:, j].fillna(last_value)
          sg_data_concat[i] = data
  sg['sg_' + str(i)] = sg_data_concat[i]

for i in Spike_Feature_Index:
  print('fillna spike_' + str(i) + '...')
  data = spike_data_concat[i].copy()
  for j in range(0, lenght-1):
      initial_data = data.iloc[:, j]
      if True in initial_data.isnull().values:
          last_value = initial_data[np.where(
              initial_data.isnull().values == True)[0][0]-1]
          data.iloc[:, j] = spike_data_concat[i].iloc[:, j].fillna(last_value)
          spike_data_concat[i] = data
  spike['spike_' + str(i)] = spike_data_concat[i]

print('cost time:', time.time() - s)

print(sg.keys())
print(spike.keys())

# np.save()
sg_feture = list(sg.keys())
spike_feture = list(spike.keys())

print('cost time:', time.time() - s)


# %%
# Calculate B's, C's, D's values at A = 0.001, 0.002, 0.003.....
s = time.time()
for i in spike_feture:
    a = np.transpose(spike[i].values).tolist()
    for j in range(len(a)):
        index = range(3, len(a[j])-1, 5)
        value = [(a[j][k-1] + a[j][k])/2 for k in index]
        a[j] = np.insert(a[j], index, value)
    spike[i] =  np.array(a)
    print(str(i) + str(spike[i].shape))
print('cost time: ', time.time() - s)

s = time.time()
for i in spike_feture:
    a = np.transpose(spike[i])
    df = pd.DataFrame(a)
    select_index = [i for i in range(0, spike[i].shape[1], 3)]
    spike[i] = df.iloc[select_index, :].reset_index(drop=True)
    print(str(i) + str(spike[i].shape))
print('cost time: ', time.time() - s)

# %%
#Calculate the difference in time of 0.001s
s = time.time()
for i in sg_feture[1:len(sg_feture)-1]:
    a = np.transpose(sg[i]).values
    CISP = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in a]
    for j in range(len(CISP)):
        CISP[j] = np.insert(CISP[j], 0, 0)
    sg[i] = CISP

for i in spike_feture[1:]:
    a = np.transpose(spike[i]).values
    CISP = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in a]
    for j in range(len(CISP)):
        CISP[j] = np.insert(CISP[j], 0, 0)
    spike[i] = CISP
print('cost time: ', time.time() - s)

# %%
#Save featrue as .npy
for i in sg_feture:
    if i != 'sg_A' and i != 'sg_I':
        sg[i] = np.transpose(sg[i])
    print(str(i) + str(sg[i].shape))
    np.save(i, sg[i])
for i in spike_feture:
    if i != 'spike_A':
        spike[i] = np.transpose(spike[i])
    print(str(i) + str(spike[i].shape))
    np.save(i, spike[i])


