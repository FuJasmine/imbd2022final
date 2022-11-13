#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras import backend as K
from keras.layers import Reshape


# In[16]:


sg_B = np.load('sg_B.npy')
sg_C = np.load('sg_C.npy')
sg_D = np.load('sg_D.npy')
sg_E = np.load('sg_E.npy')
sg_F = np.load('sg_F.npy')
sg_G = np.load('sg_G.npy')
sg_H = np.load('sg_H.npy')
sg_I = np.load('sg_I.npy')
BDF_distance = np.load('BDF_distance.npy')


spike_A = np.load('spike_A.npy')
spike_B = np.load('spike_B.npy')
spike_C = np.load('spike_C.npy')
spike_D = np.load('spike_D.npy')
spike_abs_B = np.load('spike_abs_B.npy')
spike_abs_C = np.load('spike_abs_C.npy')
spike_abs_D = np.load('spike_abs_D.npy')

spike_B_lower_noise = np.load('spike_B_lower_noise.npy')
spike_C_lower_noise = np.load('spike_C_lower_noise.npy')
spike_abs_B_lower_noise = np.load('spike_abs_B_lower_noise.npy')
spike_abs_C_lower_noise = np.load('spike_abs_C_lower_noise.npy')

BCD_distance = np.load('BCD_distance.npy')
BCD_abs_distance = np.load('BCD_abs_distance.npy')

Output = pd.read_csv('train1/00_Wear_data.csv').loc[:, 'MaxWear'].values


# In[17]:


spike_B_sum = spike_B.sum(axis=1)
spike_C_sum = spike_C.sum(axis=1)
spike_D_sum = spike_D.sum(axis=1)

spike_B_lower_noise_sum = spike_B_lower_noise.sum(axis=1)
spike_C_lower_noise_sum = spike_C_lower_noise.sum(axis=1)

df2 = pd.DataFrame()
integrated_spike_B = [ sum(abs(spike_B_sum[:i]))for i in range(1, len(spike_B_sum)+1)]
df2['integrated_spike_B'] = pd.Series(integrated_spike_B)

integrated_spike_C = [ sum(abs(spike_C_sum[:i]))for i in range(1, len(spike_C_sum)+1)]
df2['integrated_spike_C'] = pd.Series(integrated_spike_C)

integrated_spike_D = [ sum(abs(spike_D_sum[:i]))for i in range(1, len(spike_D_sum)+1)]
df2['integrated_spike_D'] = pd.Series(integrated_spike_D)
""""""
integrated_spike_B_lower_noise = [ sum(abs(spike_B_lower_noise_sum[:i]))for i in range(1, len(spike_B_lower_noise_sum)+1)]
df2['integrated_spike_B_lower_noise'] = pd.Series(integrated_spike_B_lower_noise)

integrated_spike_C_lower_noise = [ sum(abs(spike_C_lower_noise_sum[:i]))for i in range(1, len(spike_C_lower_noise_sum)+1)]
df2['integrated_spike_C_lower_noise'] = pd.Series(integrated_spike_C_lower_noise)

#df2['integrated_spike_B ^2'] = pow(df2['integrated_spike_B'], 2)
#df2['integrated_spike_C ^2'] = pow(df2['integrated_spike_C'], 2)
#df2['integrated_spike_D ^2'] = pow(df2['integrated_spike_D'], 2)
#df2['integrated_spike_B_lower_noise ^2'] = pow(df2['integrated_spike_B_lower_noise'], 2)
#df2['integrated_spike_C_lower_noise ^2'] = pow(df2['integrated_spike_C_lower_noise'], 2)

#df2['integrated_spike_B ^3'] = pow(df2['integrated_spike_B'], 3)
#df2['integrated_spike_C ^3'] = pow(df2['integrated_spike_C'], 3)
#df2['integrated_spike_D ^3'] = pow(df2['integrated_spike_D'], 3)
#df2['integrated_spike_B_lower_noise ^3'] = pow(df2['integrated_spike_B_lower_noise'], 3)
#df2['integrated_spike_C_lower_noise ^3'] = pow(df2['integrated_spike_C_lower_noise'], 3)

#df2['integrated_spike'] = df2['integrated_spike_B'] + df2['integrated_spike_C'] + df2['integrated_spike_D']
#df2['integrated_spike ^2'] = pow(df2['integrated_spike'], 2)
#df2['integrated_spike ^3'] = pow(df2['integrated_spike'], 3)

df2['Output'] = pd.Series(Output)

columns = df2.columns
transformer = StandardScaler()
corr = pd.DataFrame(transformer.fit_transform(df2), columns = columns).corr()
print(corr.iloc[-1, :])


#plt.subplots(figsize=(15, 12))
#sns.heatmap(corr, vmax=0.9, cmap='Blues', annot=True, square=True)


# In[18]:


print(df2.columns)
Model = df2.values
Input = Model[:, :-1]
output = Model[:, -1]
output = np.reshape(output, (-1, 1))

Input_shape = Input.shape[1]

print('Input layer 0: ',Input[0,:10])
print('Output layer ~ 10: ', output[:10])

Input_transformer = MaxAbsScaler()
Output_transformer = StandardScaler()

Input = Input_transformer.fit_transform(Input)
Output = Output_transformer.fit_transform(output)

print('Input layer 0: ',Input[0,:10])

input_train, input_test, output_train, output_test = train_test_split(Input, output, test_size=0.1, random_state=42)
print('input_train.shape:\t', input_train.shape)
print('input_test.shape:\t', input_test.shape)
print('output_train.shape:\t', output_train.shape)
print('output_test.shape:\t', output_test.shape)

input_train = tf.convert_to_tensor(input_train)
input_test = tf.convert_to_tensor(input_test)
output_train = tf.convert_to_tensor(output_train)
output_test = tf.convert_to_tensor(output_test)



# 256 B D0.2 128 B D0.2 64 B D0.2 32 B D0.2 4 1
# 256 B D0.2 4 1
# 256 16 4 1
# 256 16 4 2 1

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=Input_shape))
#model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
#model.add(Dense(2, activation='relu'))
model.add(Dense(1))




model.summary()


# In[22]:


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


learning_rate = 0.0005    # 0.003 --> 0.0003    0.005 --> 0.0005
batch_size = 15
epochs = 500            


# Adam RMSprop
model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=rmse)

start = time.time()
history = model.fit(input_train, output_train,
                    validation_data=(input_test, output_test), batch_size=batch_size, epochs=epochs, verbose=1)
end = time.time()

print('\n\n')
print('Training cost time:\t', end - start, 's')
print('\n\n')


# In[23]:


def numpy_rmse(actual, predict):
    return pow(np.mean(pow(actual - predict, 2)), 0.5)

test_predict = model.predict(input_test)
test_predict_actual = output_test.numpy()

test_predict = np.array(test_predict)
print(list(test_predict))

test_predict = Output_transformer.inverse_transform(test_predict)
test_predict_actual = Output_transformer.inverse_transform(test_predict_actual)

test_RMSE = numpy_rmse(test_predict_actual, test_predict)
print('Test: ')
print('Test RMSE:\t', test_RMSE)
#print('Test actual:\t', test_predict_actual.T)
#print('Test predict:\t', test_predict.T)
print()

###############################################################

train_predict = model.predict(input_train)
train_predict_actual = output_train.numpy()

train_predict = Output_transformer.inverse_transform(train_predict)
train_predict_actual = Output_transformer.inverse_transform(train_predict_actual)

train_RMSE = numpy_rmse(train_predict_actual, train_predict)
print('\n\n\n')
print('Train: ')
print('Train RMSE:\t', train_RMSE)
#print('Train actual:\t', train_predict_actual.T)
#print('Train predict:\t', train_predict.T)


##############################################################

total_predict = model.predict(Input)
total_predict_actual = output

total_predict = Output_transformer.inverse_transform(total_predict)
total_predict_actual = Output_transformer.inverse_transform(total_predict_actual)

total_RMSE = numpy_rmse(total_predict_actual, total_predict)
print('\n\n\n')
print('Total: ')
print('Total RMSE:\t', total_RMSE)
#print('Total actual:\t', total_predict_actual.T)
#print('Total predict:\t', total_predict.T)

"""
print('Total actual:')
for i in total_predict_actual:
        print(i[0], ', ', end='')
print('\nTotal predict:')
for i in total_predict:
        print(i[0], ', ', end='')
"""


# In[24]:


fig, ax1 = plt.subplots(figsize=(100, 30))
plt.plot(total_predict_actual, linewidth=10.0)
plt.plot(total_predict, color='orange', linewidth=10.0)


# In[ ]:





# In[ ]:




