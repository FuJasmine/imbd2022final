# %%
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

# %%
path = 'train_data/'

sg_B = np.load(path + 'sg_B.npy')
sg_C = np.load(path + 'sg_C.npy')
sg_D = np.load(path + 'sg_D.npy')
sg_E = np.load(path + 'sg_E.npy')
sg_F = np.load(path + 'sg_F.npy')
sg_G = np.load(path + 'sg_G.npy')
sg_H = np.load(path + 'sg_H.npy')
sg_I = np.load(path + 'sg_I.npy')
BDF_distance = np.load(path + 'BDF_distance.npy')


spike_A = np.load(path + 'spike_A.npy')
spike_B = np.load(path + 'spike_B.npy')
spike_C = np.load(path + 'spike_C.npy')
spike_D = np.load(path + 'spike_D.npy')
spike_abs_B = np.load(path + 'spike_abs_B.npy')
spike_abs_C = np.load(path + 'spike_abs_C.npy')
spike_abs_D = np.load(path + 'spike_abs_D.npy')

spike_B_lower_noise = np.load(path + 'spike_B_lower_noise.npy')
spike_C_lower_noise = np.load(path + 'spike_C_lower_noise.npy')
spike_abs_B_lower_noise = np.load(path + 'spike_abs_B_lower_noise.npy')
spike_abs_C_lower_noise = np.load(path + 'spike_abs_C_lower_noise.npy')

BCD_distance = np.load(path + 'BCD_distance.npy')
BCD_abs_distance = np.load(path + 'BCD_abs_distance.npy')


output1 = pd.read_csv('train1/00_Wear_data.csv').loc[:, 'MaxWear']
output2 = pd.read_csv('train2/00_Wear_data.csv').loc[:, 'MaxWear']
Output = pd.concat([output1, output2], axis=0).values


# %%
spike_B_sum = spike_B.sum(axis=1)
spike_C_sum = spike_C.sum(axis=1)
spike_D_sum = spike_D.sum(axis=1)

spike_B_lower_noise_sum = spike_B_lower_noise.sum(axis=1)
spike_C_lower_noise_sum = spike_C_lower_noise.sum(axis=1)

# %%
train1_len = 46

df2 = pd.DataFrame()
integrated_spike_B1 = [sum(abs(spike_B_sum[:i]))
                       for i in range(1, train1_len+1)]
integrated_spike_B2 = [sum(abs(spike_B_sum[train1_len+1:i]))
                       for i in range(train1_len+1, len(spike_B_sum)+1)]
integrated_spike_B = integrated_spike_B1 + integrated_spike_B2
df2['integrated_spike_B'] = pd.Series(integrated_spike_B)


integrated_spike_C1 = [sum(abs(spike_C_sum[:i]))
                       for i in range(1, train1_len+1)]
integrated_spike_C2 = [sum(abs(spike_C_sum[train1_len+1:i]))
                       for i in range(train1_len+1, len(spike_C_sum)+1)]
integrated_spike_C = integrated_spike_C1 + integrated_spike_C2
df2['integrated_spike_C'] = pd.Series(integrated_spike_C)

integrated_spike_D1 = [sum(abs(spike_D_sum[:i]))
                       for i in range(1, train1_len+1)]
integrated_spike_D2 = [sum(abs(spike_D_sum[train1_len+1:i]))
                       for i in range(train1_len+1, len(spike_D_sum)+1)]
integrated_spike_D = integrated_spike_D1 + integrated_spike_D2
df2['integrated_spike_D'] = pd.Series(integrated_spike_D)

integrated_spike_B_lower_noise1 = [sum(abs(
    spike_B_lower_noise_sum[:i]))for i in range(1, train1_len+1)]
integrated_spike_B_lower_noise2 = [sum(abs(
    spike_B_lower_noise_sum[train1_len+1:i]))for i in range(train1_len+1, len(spike_B_lower_noise_sum)+1)]
integrated_spike_B_lower_noise = integrated_spike_B_lower_noise1 + \
    integrated_spike_B_lower_noise2
df2['integrated_spike_B_lower_noise'] = pd.Series(
    integrated_spike_B_lower_noise)

integrated_spike_C_lower_noise1 = [sum(abs(
    spike_C_lower_noise_sum[:i]))for i in range(1, train1_len+1)]
integrated_spike_C_lower_noise2 = [sum(abs(
    spike_C_lower_noise_sum[i-1:i]))for i in range(train1_len+1, len(spike_C_lower_noise_sum)+1)]
integrated_spike_C_lower_noise = integrated_spike_C_lower_noise1 + \
    integrated_spike_C_lower_noise2
df2['integrated_spike_C_lower_noise'] = pd.Series(
    integrated_spike_C_lower_noise)

df2['Output'] = pd.Series(Output)

columns = df2.columns
transformer = StandardScaler()
corr = pd.DataFrame(transformer.fit_transform(df2), columns=columns).corr()
print(corr.iloc[-1, :])


# %%
print(df2.columns)
Model = df2.values
Input = Model[:, :-1]
output = Model[:, -1]
output = np.reshape(output, (-1, 1))

Input_shape = Input.shape[1]

print('Input layer 0: ', Input[0, :10])
print('Output layer ~ 10: ', output[:10])

Input_transformer = MaxAbsScaler()
Output_transformer = StandardScaler()

Input = Input_transformer.fit_transform(Input)
Output = Output_transformer.fit_transform(output)

print('Input layer 0: ', Input[0, :10])

input_train, input_test, output_train, output_test = train_test_split(
    Input, output, test_size=0.1, random_state=42)
print('input_train.shape:\t', input_train.shape)
print('input_test.shape:\t', input_test.shape)
print('output_train.shape:\t', output_train.shape)
print('output_test.shape:\t', output_test.shape)

input_train = tf.convert_to_tensor(input_train)
input_test = tf.convert_to_tensor(input_test)
output_train = tf.convert_to_tensor(output_train)
output_test = tf.convert_to_tensor(output_test)


# %%
model = Sequential()

model.add(Dense(128, activation='relu', input_dim=Input_shape))
#model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
#model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()

# %%
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


learning_rate = 0.0005
batch_size = 15
epochs = 500


# Adam RMSprop
model.compile(optimizer=tf.optimizers.Adam(
    learning_rate=learning_rate), loss=rmse)

start = time.time()
history = model.fit(input_train, output_train,
                    validation_data=(input_test, output_test), batch_size=batch_size, epochs=epochs, verbose=1)
end = time.time()

print('\n\n')
print('Training cost time:\t', end - start, 's')
print('\n\n')


# %%
def numpy_rmse(actual, predict):
    return pow(np.mean(pow(actual - predict, 2)), 0.5)


test_predict = model.predict(input_test)
test_predict_actual = output_test.numpy()

test_predict = np.array(test_predict)

test_predict = Output_transformer.inverse_transform(test_predict)
test_predict_actual = Output_transformer.inverse_transform(test_predict_actual)

test_RMSE = numpy_rmse(test_predict_actual, test_predict)
print('Test: ')
print('Test RMSE:\t', test_RMSE)


# %%
train_predict = model.predict(input_train)
train_predict_actual = output_train.numpy()

train_predict = Output_transformer.inverse_transform(train_predict)
train_predict_actual = Output_transformer.inverse_transform(
    train_predict_actual)

train_RMSE = numpy_rmse(train_predict_actual, train_predict)
print('\n\n\n')
print('Train: ')
print('Train RMSE:\t', train_RMSE)


# %%
total_predict = model.predict(Input)
total_predict_actual = output

total_predict = Output_transformer.inverse_transform(total_predict)
total_predict_actual = Output_transformer.inverse_transform(
    total_predict_actual)

total_RMSE = numpy_rmse(total_predict_actual, total_predict)
print('\n\n\n')
print('Total: ')
print('Total RMSE:\t', total_RMSE)


# %%
fig, ax1 = plt.subplots(figsize=(100, 30))
plt.plot(total_predict_actual, linewidth=10.0)
plt.plot(total_predict, color='orange', linewidth=10.0)


# %%
path = 'test_data/'

sg_B = np.load(path + 'sg_B.npy')
sg_C = np.load(path + 'sg_C.npy')
sg_D = np.load(path + 'sg_D.npy')
sg_E = np.load(path + 'sg_E.npy')
sg_F = np.load(path + 'sg_F.npy')
sg_G = np.load(path + 'sg_G.npy')
sg_H = np.load(path + 'sg_H.npy')
sg_I = np.load(path + 'sg_I.npy')
BDF_distance = np.load(path + 'BDF_distance.npy')


spike_A = np.load(path + 'spike_A.npy')
spike_B = np.load(path + 'spike_B.npy')
spike_C = np.load(path + 'spike_C.npy')
spike_D = np.load(path + 'spike_D.npy')
spike_abs_B = np.load(path + 'spike_abs_B.npy')
spike_abs_C = np.load(path + 'spike_abs_C.npy')
spike_abs_D = np.load(path + 'spike_abs_D.npy')

spike_B_lower_noise = np.load(path + 'spike_B_lower_noise.npy')
spike_C_lower_noise = np.load(path + 'spike_C_lower_noise.npy')
spike_abs_B_lower_noise = np.load(path + 'spike_abs_B_lower_noise.npy')
spike_abs_C_lower_noise = np.load(path + 'spike_abs_C_lower_noise.npy')

BCD_distance = np.load(path + 'BCD_distance.npy')
BCD_abs_distance = np.load(path + 'BCD_abs_distance.npy')


# %%
spike_B_sum = spike_B.sum(axis=1)
spike_C_sum = spike_C.sum(axis=1)
spike_D_sum = spike_D.sum(axis=1)

spike_B_lower_noise_sum = spike_B_lower_noise.sum(axis=1)
spike_C_lower_noise_sum = spike_C_lower_noise.sum(axis=1)

df2 = pd.DataFrame()
integrated_spike_B = [sum(abs(spike_B_sum[:i]))
                      for i in range(1, len(spike_B_sum)+1)]
df2['integrated_spike_B'] = pd.Series(integrated_spike_B)

integrated_spike_C = [sum(abs(spike_C_sum[:i]))
                      for i in range(1, len(spike_C_sum)+1)]
df2['integrated_spike_C'] = pd.Series(integrated_spike_C)

integrated_spike_D = [sum(abs(spike_D_sum[:i]))
                      for i in range(1, len(spike_D_sum)+1)]
df2['integrated_spike_D'] = pd.Series(integrated_spike_D)

integrated_spike_B_lower_noise = [sum(abs(
    spike_B_lower_noise_sum[:i]))for i in range(1, len(spike_B_lower_noise_sum)+1)]
df2['integrated_spike_B_lower_noise'] = pd.Series(
    integrated_spike_B_lower_noise)

integrated_spike_C_lower_noise = [sum(abs(
    spike_C_lower_noise_sum[:i]))for i in range(1, len(spike_C_lower_noise_sum)+1)]
df2['integrated_spike_C_lower_noise'] = pd.Series(
    integrated_spike_C_lower_noise)

columns = df2.columns
transformer = StandardScaler()
corr = pd.DataFrame(transformer.fit_transform(df2), columns=columns).corr()
print(corr.iloc[-1, :])

# %%
print(df2.columns)
Model = df2.values
Input = Model

Input_shape = Input.shape[1]
print('Input layer 0: ', Input[0, :10])

Input_transformer = MaxAbsScaler()
Input = Input_transformer.fit_transform(Input)
print('Input layer 0: ', Input[0, :10])


# %%
total_predict = model.predict(Input)
total_predict = Output_transformer.inverse_transform(total_predict)
print('Total predict:\t', total_predict.T)

# %%
save_name = '111052_projectB_ans.csv'  

# load_name = '111052_projectB_ans.csv'
# df3 = pd.read_csv(load_name)
# df3.loc[:, 'MaxWear'] = total_predict
# df3.to_csv(save_name, index=False)

df3 = pd.DataFrame()
index = np.arange(1, 26)
df3['Index'] = pd.Series(index)
df3['MaxWear'] = total_predict
df3.to_csv(save_name, index=False)
print('Saved to ', save_name)



