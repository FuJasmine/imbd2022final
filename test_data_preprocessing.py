# %%
import pandas as pd
import numpy as np
from numpy import insert
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import os

# %%
# Read file and seperate each feature to do data preprocessing

print('Reading sg...')
sg = [pd.read_csv('test/' + str(i) + '_sg.csv') for i in range(1, 26)]
print('Reading spike...')
spike = [pd.read_csv('test/' + str(i) + '_spike.csv') for i in range(1, 26)]

print('\n\n')
print('Seperate each feature to do data preprocessing')
print('-*-*-*' * 18 + '-')

s = time.time()
print('A...', end='')
sg_A = [i.values.T[0] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
sg_B = [i.values.T[1] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
sg_C = [i.values.T[2] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
sg_D = [i.values.T[3] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('E...', end='')
sg_E = [i.values.T[4] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('F...', end='')
sg_F = [i.values.T[5] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('G...', end='')
sg_G = [i.values.T[6] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('H...', end='')
sg_H = [i.values.T[7] for i in sg]
print('cost time: ', time.time() - s)

s = time.time()
print('I...', end='')
sg_I = [i.values.T[8] for i in sg]
print('cost time: ', time.time() - s)

################################################################
print()
s = time.time()
print('A...', end='')
spike_A = [i.values.T[0] for i in spike]
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
spike_B = [i.values.T[1] for i in spike]
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
spike_C = [i.values.T[2] for i in spike]
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
spike_D = [i.values.T[3] for i in spike]
print('cost time: ', time.time() - s)


# %%
# Show the size of our data
print('\n\n')
print('-*-*-*' * 18 + '-')

print('sg_A|\t\t1:', sg_A[0].size, '\t24:', sg_A[23].size,
      '\t25:', sg_A[24].size)
print('--------'*10)
print('sg_B|\t\t1:', sg_B[0].size, '\t24:', sg_B[23].size,
      '\t25:', sg_B[24].size)
print('sg_C|\t\t1:', sg_C[0].size, '\t24:', sg_C[23].size,
      '\t25:', sg_C[24].size)
print('sg_D|\t\t1:', sg_D[0].size, '\t24:', sg_D[23].size,
      '\t25:', sg_D[24].size)
print('sg_E|\t\t1:', sg_E[0].size, '\t24:', sg_E[23].size,
      '\t25:', sg_E[24].size)
print('sg_F|\t\t1:', sg_F[0].size, '\t24:', sg_F[23].size,
      '\t25:', sg_F[24].size)
print('sg_G|\t\t1:', sg_G[0].size, '\t24:', sg_G[23].size,
      '\t25:', sg_G[24].size)
print('sg_H|\t\t1:', sg_H[0].size, '\t24:', sg_H[23].size,
      '\t25:', sg_H[24].size)
print('sg_I|\t\t1:', sg_I[0].size, '\t24:', sg_I[23].size,
      '\t25:', sg_I[24].size)


print('\n')
print('spike_A|\t1:', spike_A[0].size, '\t24:', spike_A[23].size,
      '\t25:', spike_A[24].size)
print('--------'*10)
print('spike_B|\t1:', spike_B[0].size, '\t24:', spike_B[23].size,
      '\t25:', spike_B[24].size)
print('spike_C|\t1:', spike_C[0].size, '\t24:', spike_C[23].size,
      '\t25:', spike_C[24].size)
print('spike_D|\t1:', spike_D[0].size, '\t24:', spike_D[23].size,
      '\t25:', spike_D[24].size)


# %%
# If the number of data in one column is smaller than maximum, we'll use the last value of that column
# to fill in to get a same dimension in one feature(column)

# ----sg----
max_num = max([i.size for i in sg_A])
print('\n\n')
print('Fill in value to get same dimension in one feature')
print('-*-*-*' * 18 + '-')

s = time.time()
print('B...', end='')
for i in range(len(sg_B)):
    if sg_B[i].size < max_num:
        sg_B[i] = np.append(
            sg_B[i], [[sg_B[i][-1]] * (max_num - sg_B[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
for i in range(len(sg_C)):
    if sg_C[i].size < max_num:
        sg_C[i] = np.append(
            sg_C[i], [[sg_C[i][-1]] * (max_num - sg_C[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
for i in range(len(sg_D)):
    if sg_D[i].size < max_num:
        sg_D[i] = np.append(
            sg_D[i], [[sg_D[i][-1]] * (max_num - sg_D[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('E...', end='')
for i in range(len(sg_E)):
    if sg_E[i].size < max_num:
        sg_E[i] = np.append(
            sg_E[i], [[sg_E[i][-1]] * (max_num - sg_E[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('F...', end='')
for i in range(len(sg_F)):
    if sg_F[i].size < max_num:
        sg_F[i] = np.append(
            sg_F[i], [[sg_F[i][-1]] * (max_num - sg_F[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('G...', end='')
for i in range(len(sg_G)):
    if sg_G[i].size < max_num:
        sg_G[i] = np.append(
            sg_G[i], [[sg_G[i][-1]] * (max_num - sg_G[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('H...', end='')
for i in range(len(sg_H)):
    if sg_H[i].size < max_num:
        sg_H[i] = np.append(
            sg_H[i], [[sg_H[i][-1]] * (max_num - sg_H[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('I...', end='')
for i in range(len(sg_I)):
    if sg_I[i].size < max_num:
        sg_I[i] = np.append(
            sg_I[i], [[sg_I[i][-1]] * (max_num - sg_I[i].size)])
print('cost time: ', time.time() - s)


print('--------'*10)
# ----spike----
max_num = max([i.size for i in spike_A])
s = time.time()
print('A...', end='')
for i in range(len(spike_A)):
    if spike_A[i].size < max_num:
        spike_A[i] = np.append(
            spike_A[i], [[spike_A[i][-1]] * (max_num - spike_A[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
for i in range(len(spike_B)):
    if spike_B[i].size < max_num:
        spike_B[i] = np.append(
            spike_B[i], [[spike_B[i][-1]] * (max_num - spike_B[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
for i in range(len(spike_C)):
    if spike_C[i].size < max_num:
        spike_C[i] = np.append(
            spike_C[i], [[spike_C[i][-1]] * (max_num - spike_C[i].size)])
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
for i in range(len(spike_D)):
    if spike_D[i].size < max_num:
        spike_D[i] = np.append(
            spike_D[i], [[spike_D[i][-1]] * (max_num - spike_D[i].size)])
print('cost time: ', time.time() - s)


# %%
# Show the size of our data
print('\n\n')
print('-*-*-*' * 18 + '-')

print('sg_A|\t\t1:', sg_A[0].size, '\t24:', sg_A[23].size,
      '\t25:', sg_A[24].size)
print('--------'*10)
print('sg_B|\t\t1:', sg_B[0].size, '\t24:', sg_B[23].size,
      '\t25:', sg_B[24].size)
print('sg_C|\t\t1:', sg_C[0].size, '\t24:', sg_C[23].size,
      '\t25:', sg_C[24].size)
print('sg_D|\t\t1:', sg_D[0].size, '\t24:', sg_D[23].size,
      '\t25:', sg_D[24].size)
print('sg_E|\t\t1:', sg_E[0].size, '\t24:', sg_E[23].size,
      '\t25:', sg_E[24].size)
print('sg_F|\t\t1:', sg_F[0].size, '\t24:', sg_F[23].size,
      '\t25:', sg_F[24].size)
print('sg_G|\t\t1:', sg_G[0].size, '\t24:', sg_G[23].size,
      '\t25:', sg_G[24].size)
print('sg_H|\t\t1:', sg_H[0].size, '\t24:', sg_H[23].size,
      '\t25:', sg_H[24].size)
print('sg_I|\t\t1:', sg_I[0].size, '\t24:', sg_I[23].size,
      '\t25:', sg_I[24].size)


print('\n')
print('spike_A|\t1:', spike_A[0].size, '\t24:', spike_A[23].size,
      '\t25:', spike_A[24].size)
print('--------'*10)
print('spike_B|\t1:', spike_B[0].size, '\t24:', spike_B[23].size,
      '\t25:', spike_B[24].size)
print('spike_C|\t1:', spike_C[0].size, '\t24:', spike_C[23].size,
      '\t25:', spike_C[24].size)
print('spike_D|\t1:', spike_D[0].size, '\t24:', spike_D[23].size,
      '\t25:', spike_D[24].size)


# %%
# Change in sg --> change the data from absolute coordination to relative coordination (distance in each timestep)
print('\n\n')
print('Value change in sg...')
print('-*-*-*' * 18 + '-')

s = time.time()
print('B...', end='')
CISG_B = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_B]
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
CISG_C = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_C]
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
CISG_D = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_D]
print('cost time: ', time.time() - s)

s = time.time()
print('E...', end='')
CISG_E = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_E]
print('cost time: ', time.time() - s)

s = time.time()
print('F...', end='')
CISG_F = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_F]
print('cost time: ', time.time() - s)

s = time.time()
print('G...', end='')
CISG_G = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_G]
print('cost time: ', time.time() - s)

s = time.time()
print('H...', end='')
CISG_H = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in sg_H]
print('cost time: ', time.time() - s)
# sg_I

for i in range(len(CISG_B)):CISG_B[i] = np.insert(CISG_B[i], 0, 0)
for i in range(len(CISG_C)):CISG_C[i] = np.insert(CISG_C[i], 0, 0)
for i in range(len(CISG_D)):CISG_D[i] = np.insert(CISG_D[i], 0, 0)
for i in range(len(CISG_E)):CISG_E[i] = np.insert(CISG_E[i], 0, 0)
for i in range(len(CISG_F)):CISG_F[i] = np.insert(CISG_F[i], 0, 0)
for i in range(len(CISG_G)):CISG_G[i] = np.insert(CISG_G[i], 0, 0)
for i in range(len(CISG_H)):CISG_H[i] = np.insert(CISG_H[i], 0, 0)

# %%
# Calculate B's, C's, D's values at A = 0.001, 0.002, 0.003.....
print('\n\n')
print("Calculate B's, C's, D's values at A = 0.001, 0.002, 0.003...")
print('-*-*-*' * 18 + '-')

A = copy.deepcopy(spike_A)
B = copy.deepcopy(spike_B)
C = copy.deepcopy(spike_C)
D = copy.deepcopy(spike_D)

s = time.time()
print('A...', end='')
for i in range(len(A)):
    index = range(3, spike_A[i].size-1, 5)
    value = [(A[i][j-1] + A[i][j])/2 for j in index]
    A[i] = insert(A[i], index, value)
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
for i in range(len(B)):
    index = range(3, spike_B[i].size-1, 5)
    value = [(B[i][j-1] + B[i][j])/2 for j in index]
    B[i] = insert(B[i], index, value)
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
for i in range(len(C)):
    index = range(3, spike_C[i].size-1, 5)
    value = [(C[i][j-1] + C[i][j])/2 for j in index]
    C[i] = insert(C[i], index, value)
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
for i in range(len(D)):
    index = range(3, spike_D[i].size-1, 5)
    value = [(D[i][j-1] + D[i][j])/2 for j in index]
    D[i] = insert(D[i], index, value)
print('cost time: ', time.time() - s)


# %%
# Change in spike --> change the data from absolute coordination to relative coordination (distance in each timestep)
print('\n\n')
print("Value change in spike...")
print('-*-*-*' * 18 + '-')

s = time.time()
print('A...', end='')
CISP_A = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in A]
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
CISP_B = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in B]
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
CISP_C = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in C]
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
CISP_D = [np.array([i[j]-i[j-1] for j in range(1, len(i))]) for i in D]
print('cost time: ', time.time() - s)

for i in range(len(CISP_A)):
    CISP_A[i] = np.insert(CISP_A[i], 0, 0)
for i in range(len(CISP_B)):
    CISP_B[i] = np.insert(CISP_B[i], 0, 0)
for i in range(len(CISP_C)):
    CISP_C[i] = np.insert(CISP_C[i], 0, 0)
for i in range(len(CISP_D)):
    CISP_D[i] = np.insert(CISP_D[i], 0, 0)

print()
print('CISP_A[0]:\t', CISP_A[0])
print('CISP_B[0]:\t', CISP_B[0])
print('CISP_C[0]:\t', CISP_C[0])
print('CISP_D[0]:\t', CISP_D[0])


# %%
# Sum up the distance that process during one timestep 0.001
# final_abs_: total path
# fianl_: distance
print('\n\n')
print('Sum up the distance that process during one timestep 0.001')
print('-*-*-*' * 18 + '-')

s = time.time()
""""""
print('A...', end='')
final_abs_A = [[abs(CISP_A[j][i*3+1:i*3+4]).sum()
                for i in range(0, CISG_B[j].size-1)] for j in range(len(CISP_A))]
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
final_abs_B = [[abs(CISP_B[j][i*3+1:i*3+4]).sum()
                for i in range(0, CISG_B[j].size-1)] for j in range(len(CISP_B))]
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
final_abs_C = [[abs(CISP_C[j][i*3+1:i*3+4]).sum()
                for i in range(0, CISG_B[j].size-1)] for j in range(len(CISP_C))]
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
final_abs_D = [[abs(CISP_D[j][i*3+1:i*3+4]).sum()
                for i in range(0, CISG_B[j].size-1)] for j in range(len(CISP_D))]
print('cost time: ', time.time() - s)


""""""
s = time.time()
print('A...', end='')
final_A = [[CISP_A[j][i*3+1:i*3+4].sum() for i in range(0, CISG_B[j].size-1)]
           for j in range(len(CISP_A))]
print('cost time: ', time.time() - s)

s = time.time()
print('B...', end='')
final_B = [[CISP_B[j][i*3+1:i*3+4].sum() for i in range(0, CISG_B[j].size-1)]
           for j in range(len(CISP_B))]
print('cost time: ', time.time() - s)

s = time.time()
print('C...', end='')
final_C = [[CISP_C[j][i*3+1:i*3+4].sum() for i in range(0, CISG_B[j].size-1)]
           for j in range(len(CISP_C))]
print('cost time: ', time.time() - s)

s = time.time()
print('D...', end='')
final_D = [[CISP_D[j][i*3+1:i*3+4].sum() for i in range(0, CISG_B[j].size-1)]
           for j in range(len(CISP_D))]
print('cost time: ', time.time() - s)


for i in final_abs_A:
    i = np.array(i.insert(0, 0))
for i in final_abs_B:
    i = np.array(i.insert(0, 0))
for i in final_abs_C:
    i = np.array(i.insert(0, 0))
for i in final_abs_D:
    i = np.array(i.insert(0, 0))
for i in final_A:
    i = np.array(i.insert(0, 0))
for i in final_B:
    i = np.array(i.insert(0, 0))
for i in final_C:
    i = np.array(i.insert(0, 0))
for i in final_D:
    i = np.array(i.insert(0, 0))

final_abs_A = np.array(final_abs_A)
final_abs_B = np.array(final_abs_B)
final_abs_C = np.array(final_abs_C)
final_abs_D = np.array(final_abs_D)
final_A = np.array(final_A)
final_B = np.array(final_B)
final_C = np.array(final_C)
final_D = np.array(final_D)

s = time.time()
print('BCD abs Distance...', end='')
BCD_abs_distance = [[pow(pow(final_abs_B[j][i], 2) + pow(final_abs_C[j][i], 2)
                     + pow(final_abs_D[j][i], 2), 0.5) for i in range(len(final_abs_A[j]))]
                    for j in range(len(final_abs_A))]
print('cost time: ', time.time() - s)
BCD_abs_distance = np.array(BCD_abs_distance)

s = time.time()
print('BCD Distance...', end='')
BCD_distance = [[pow(pow(final_B[j][i], 2) + pow(final_C[j][i], 2)
                     + pow(final_D[j][i], 2), 0.5) for i in range(len(final_A[j]))]
                for j in range(len(final_A))]
print('cost time: ', time.time() - s)
BCD_distance = np.array(BCD_distance)


s = time.time()
print('BDF Distance...', end='')
BDF_distance = [[pow(pow(sg_B[j][i], 2) + pow(sg_D[j][i], 2)
                     + pow(sg_F[j][i], 2), 0.5) for i in range(len(sg_B[j]))]
                for j in range(len(sg_B))]
print('cost time: ', time.time() - s)
BDF_distance = np.array(BDF_distance)


# %%
# names of variables change

sg_B = CISG_B
sg_C = CISG_C
sg_D = CISG_D
sg_E = CISG_E
sg_F = CISG_F
sg_G = CISG_G
sg_H = CISG_H
# sg_I

spike_abs_A = final_abs_A
spike_abs_B = final_abs_B
spike_abs_C = final_abs_C
spike_abs_D = final_abs_D
spike_A = final_A
spike_B = final_B
spike_C = final_C
spike_D = final_D


# %%
print('\n\n')
print('-*-*-*' * 18 + '-')

print('sg_A|\t\t1:', sg_A[0].size, '\t24:', sg_A[23].size,
      '\t25:', sg_A[24].size)
print('--------'*10)
print('sg_B|\t\t1:', sg_B[0].size, '\t24:', sg_B[23].size,
      '\t25:', sg_B[24].size)
print('sg_C|\t\t1:', sg_C[0].size, '\t24:', sg_C[23].size,
      '\t25:', sg_C[24].size)
print('sg_D|\t\t1:', sg_D[0].size, '\t24:', sg_D[23].size,
      '\t25:', sg_D[24].size)
print('sg_E|\t\t1:', sg_E[0].size, '\t24:', sg_E[23].size,
      '\t25:', sg_E[24].size)
print('sg_F|\t\t1:', sg_F[0].size, '\t24:', sg_F[23].size,
      '\t25:', sg_F[24].size)
print('sg_G|\t\t1:', sg_G[0].size, '\t24:', sg_G[23].size,
      '\t25:', sg_G[24].size)
print('sg_H|\t\t1:', sg_H[0].size, '\t24:', sg_H[23].size,
      '\t25:', sg_H[24].size)
print('sg_I|\t\t1:', sg_I[0].size, '\t24:', sg_I[23].size,
      '\t25:', sg_I[24].size)

""""""
print()
print('spike_A|\t1:', spike_A[0].size, '\t24:', spike_A[23].size,
      '\t25:', spike_A[24].size)
print('--------'*10)
print('spike_B|\t1:', spike_B[0].size, '\t24:', spike_B[23].size,
      '\t25:', spike_B[24].size)
print('spike_C|\t1:', spike_C[0].size, '\t24:', spike_C[23].size,
      '\t25:', spike_C[24].size)
print('spike_D|\t1:', spike_D[0].size, '\t24:', spike_D[23].size,
      '\t25:', spike_D[24].size)
print('BCD Distance|\t1:', BCD_distance[0].size, '\t24:', BCD_distance[23].size,
      '\t25:', BCD_distance[24].size)
"""
print('BDF Distance|\t1:', BDF_distance[0].size, '\t24:', BDF_distance[23].size, 
      '\t25:', BDF_distance[24].size, '\t46:', BDF_distance[45].size)
"""


# %%
work_B = [[1 if abs(sg_B[i][j]) >= 0.0029 and abs(sg_B[i][j]) <= 0.0041 else 0
           for j in range(len(sg_B[i]))] for i in range(len(sg_B))]
work_D = [[1 if abs(sg_D[i][j]) >= 0.0029 and abs(sg_D[i][j]) <= 0.0041 else 0
           for j in range(len(sg_D[i]))] for i in range(len(sg_D))]


spike_abs_B_lower_noise = spike_abs_B * work_B
spike_B_lower_noise = spike_B * work_B
spike_abs_C_lower_noise = spike_abs_D * work_D
spike_C_lower_noise = spike_D * work_D


# %%
path = 'test_data/'
os.makedirs(path, exist_ok=True)

sg_B = np.array(sg_B)
sg_C = np.array(sg_C)
sg_D = np.array(sg_D)
sg_E = np.array(sg_E)
sg_F = np.array(sg_F)
sg_G = np.array(sg_G)
sg_H = np.array(sg_H)
sg_I = np.array(sg_I)

spike_abs_A = np.array(spike_abs_A)
spike_abs_B = np.array(spike_abs_B)
spike_abs_C = np.array(spike_abs_C)
spike_abs_D = np.array(spike_abs_D)
spike_A = np.array(spike_A)
spike_B = np.array(spike_B)
spike_C = np.array(spike_C)
spike_D = np.array(spike_D)

spike_abs_B_lower_noise = np.array(spike_abs_B_lower_noise)
spike_B_lower_noise = np.array(spike_B_lower_noise)
spike_abs_C_lower_noise = np.array(spike_abs_C_lower_noise)
spike_C_lower_noise = np.array(spike_C_lower_noise)

BCD_distance = np.array(BCD_distance)
BCD_abs_distance = np.array(BCD_abs_distance)
BDF_distance = np.array(BDF_distance)

print('Saving test data...')

np.save( path +'sg_B', sg_B)
np.save( path +'sg_C', sg_C)
np.save( path +'sg_D', sg_D)
np.save( path +'sg_E', sg_E)
np.save( path +'sg_F', sg_F)
np.save( path +'sg_G', sg_G)
np.save( path +'sg_H', sg_H)
np.save( path +'sg_I', sg_I)

np.save( path +'spike_A', spike_A)
np.save( path +'spike_B', spike_B)
np.save( path +'spike_C', spike_C)
np.save( path +'spike_D', spike_D)
np.save( path +'spike_abs_A', spike_abs_A)
np.save( path +'spike_abs_B', spike_abs_B)
np.save( path +'spike_abs_C', spike_abs_C)
np.save( path +'spike_abs_D', spike_abs_D)

np.save( path +'spike_abs_B_lower_noise', spike_abs_B_lower_noise)
np.save( path +'spike_B_lower_noise', spike_B_lower_noise)
np.save( path +'spike_abs_C_lower_noise', spike_abs_C_lower_noise)
np.save( path +'spike_C_lower_noise', spike_C_lower_noise)

np.save( path +'BCD_distance', BCD_distance)
np.save( path +'BCD_abs_distance', BCD_abs_distance)
np.save( path +'BDF_distance', BDF_distance)


