# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

# %%
# Loading the dataset
path1 = 'train1_data/'
path2 = 'train2_data/'

sg_B1 = np.load(path1 + 'sg_B.npy')
sg_C1 = np.load(path1 + 'sg_C.npy')
sg_D1 = np.load(path1 + 'sg_D.npy')
sg_E1 = np.load(path1 + 'sg_E.npy')
sg_F1 = np.load(path1 + 'sg_F.npy')
sg_G1 = np.load(path1 + 'sg_G.npy')
sg_H1 = np.load(path1 + 'sg_H.npy')
sg_I1 = np.load(path1 + 'sg_I.npy')
BDF_distance1 = np.load(path1 + 'BDF_distance.npy')


spike_A1 = np.load(path1 + 'spike_A.npy')
spike_B1 = np.load(path1 + 'spike_B.npy')
spike_C1 = np.load(path1 + 'spike_C.npy')
spike_D1 = np.load(path1 + 'spike_D.npy')
spike_abs_B1 = np.load(path1 + 'spike_abs_B.npy')
spike_abs_C1 = np.load(path1 + 'spike_abs_C.npy')
spike_abs_D1 = np.load(path1 + 'spike_abs_D.npy')

spike_B_lower_noise1 = np.load(path1 + 'spike_B_lower_noise.npy')
spike_C_lower_noise1 = np.load(path1 + 'spike_C_lower_noise.npy')
spike_abs_B_lower_noise1 = np.load(path1 + 'spike_abs_B_lower_noise.npy')
spike_abs_C_lower_noise1 = np.load(path1 + 'spike_abs_C_lower_noise.npy')

BCD_distance1 = np.load(path1 + 'BCD_distance.npy')
BCD_abs_distance1 = np.load(path1 + 'BCD_abs_distance.npy')

sg_B2 = np.load(path2 + 'sg_B.npy')
sg_C2 = np.load(path2 + 'sg_C.npy')
sg_D2 = np.load(path2 + 'sg_D.npy')
sg_E2 = np.load(path2 + 'sg_E.npy')
sg_F2 = np.load(path2 + 'sg_F.npy')
sg_G2 = np.load(path2 + 'sg_G.npy')
sg_H2 = np.load(path2 + 'sg_H.npy')
sg_I2 = np.load(path2 + 'sg_I.npy')
BDF_distance2 = np.load(path2 + 'BDF_distance.npy')


spike_A2 = np.load(path2 + 'spike_A.npy')
spike_B2 = np.load(path2 + 'spike_B.npy')
spike_C2 = np.load(path2 + 'spike_C.npy')
spike_D2 = np.load(path2 + 'spike_D.npy')
spike_abs_B2 = np.load(path2 + 'spike_abs_B.npy')
spike_abs_C2 = np.load(path2 + 'spike_abs_C.npy')
spike_abs_D2 = np.load(path2 + 'spike_abs_D.npy')

spike_B_lower_noise2 = np.load(path2 + 'spike_B_lower_noise.npy')
spike_C_lower_noise2 = np.load(path2 + 'spike_C_lower_noise.npy')
spike_abs_B_lower_noise2 = np.load(path2 + 'spike_abs_B_lower_noise.npy')
spike_abs_C_lower_noise2 = np.load(path2 + 'spike_abs_C_lower_noise.npy')

BCD_distance2 = np.load(path2 + 'BCD_distance.npy')
BCD_abs_distance2 = np.load(path2 + 'BCD_abs_distance.npy')

# Concatenate the data
sg_B = np.concatenate((sg_B1, sg_B2), axis=0)
sg_C = np.concatenate((sg_C1, sg_C2), axis=0)
sg_D = np.concatenate((sg_D1, sg_D2), axis=0)
sg_E = np.concatenate((sg_E1, sg_E2), axis=0)
sg_F = np.concatenate((sg_F1, sg_F2), axis=0)
sg_G = np.concatenate((sg_G1, sg_G2), axis=0)
sg_H = np.concatenate((sg_H1, sg_H2), axis=0)
sg_I = np.concatenate((sg_I1, sg_I2), axis=0)
BDF_distance = np.concatenate((BDF_distance1, BDF_distance2), axis=0)


spike_A = np.concatenate((spike_A1, spike_A2), axis=0)
spike_B = np.concatenate((spike_B1, spike_B2), axis=0)
spike_C = np.concatenate((spike_C1, spike_C2), axis=0)
spike_D = np.concatenate((spike_D1, spike_D2), axis=0)
spike_abs_B = np.concatenate((spike_abs_B1, spike_abs_B2), axis=0)
spike_abs_C = np.concatenate((spike_abs_C1, spike_abs_C2), axis=0)
spike_abs_D = np.concatenate((spike_abs_D1, spike_abs_D2), axis=0)

spike_B_lower_noise = np.concatenate((spike_B_lower_noise1, spike_B_lower_noise2), axis=0)
spike_C_lower_noise = np.concatenate((spike_C_lower_noise1, spike_C_lower_noise2), axis=0)
spike_abs_B_lower_noise = np.concatenate((spike_abs_B_lower_noise1, spike_abs_B_lower_noise2), axis=0)
spike_abs_C_lower_noise = np.concatenate((spike_abs_C_lower_noise1, spike_abs_C_lower_noise2), axis=0)

BCD_distance = np.concatenate((BCD_distance1, BCD_distance2), axis=0)
BCD_abs_distance = np.concatenate((BCD_abs_distance1, BCD_abs_distance2), axis=0)

output1 = pd.read_csv('train1/00_Wear_data.csv').loc[:, 'MaxWear']
output2 = pd.read_csv('train2/00_Wear_data.csv').loc[:, 'MaxWear']
Output = pd.concat([output1, output2], axis=0).values

# %%
spike_B_sum = spike_B.sum(axis=1)
spike_C_sum = spike_C.sum(axis=1)
spike_D_sum = spike_D.sum(axis=1)

spike_B_lower_noise_sum = spike_B_lower_noise.sum(axis=1)
spike_C_lower_noise_sum = spike_C_lower_noise.sum(axis=1)

spike_B_sg_B_sum = np.array([spike_B[i]-sg_B[i] for i in range(len(spike_B))]).sum(axis=1)
spike_C_sg_D_sum = np.array([spike_C[i]-sg_D[i]
                       for i in range(len(spike_C))]).sum(axis=1)
spike_D_sg_F_sum = np.array([spike_D[i]-sg_F[i]
                       for i in range(len(spike_D))]).sum(axis=1)


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

integrated_spike_B_sg_B1 = [sum(abs(spike_B_sg_B_sum[:i]))
                            for i in range(1, train1_len+1)]
integrated_spike_B_sg_B2 = [sum(abs(spike_B_sg_B_sum[train1_len+1:i]))
                            for i in range(train1_len+1, len(spike_B_sg_B_sum)+1)]
integrated_spike_B_sg_B = integrated_spike_B_sg_B1 + integrated_spike_B_sg_B2
df2['integrated_spike_B_sg_B'] = pd.Series(integrated_spike_B_sg_B)

integrated_spike_C_sg_D1 = [sum(abs(spike_C_sg_D_sum[:i]))
                            for i in range(1, train1_len+1)]
integrated_spike_C_sg_D2 = [sum(abs(spike_C_sg_D_sum[train1_len+1:i]))
                            for i in range(train1_len+1, len(spike_C_sg_D_sum)+1)]
integrated_spike_C_sg_D = integrated_spike_C_sg_D1 + integrated_spike_C_sg_D2
df2['integrated_spike_C_sg_D'] = pd.Series(integrated_spike_C_sg_D)

integrated_spike_D_sg_F1 = [sum(abs(spike_D_sg_F_sum[:i]))
                            for i in range(1, train1_len+1)]
integrated_spike_D_sg_F2 = [sum(abs(spike_D_sg_F_sum[train1_len+1:i]))
                            for i in range(train1_len+1, len(spike_D_sg_F_sum)+1)]
integrated_spike_D_sg_F = integrated_spike_D_sg_F1 + integrated_spike_D_sg_F2
df2['integrated_spike_D_sg_F'] = pd.Series(integrated_spike_D_sg_F)


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


# %%
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
path3 = 'test_data/'

sg_B = np.load(path3 + 'sg_B.npy')
sg_C = np.load(path3 + 'sg_C.npy')
sg_D = np.load(path3 + 'sg_D.npy')
sg_E = np.load(path3 + 'sg_E.npy')
sg_F = np.load(path3 + 'sg_F.npy')
sg_G = np.load(path3 + 'sg_G.npy')
sg_H = np.load(path3 + 'sg_H.npy')
sg_I = np.load(path3 + 'sg_I.npy')
BDF_distance = np.load(path3 + 'BDF_distance.npy')


spike_A = np.load(path3 + 'spike_A.npy')
spike_B = np.load(path3 + 'spike_B.npy')
spike_C = np.load(path3 + 'spike_C.npy')
spike_D = np.load(path3 + 'spike_D.npy')
spike_abs_B = np.load(path3 + 'spike_abs_B.npy')
spike_abs_C = np.load(path3 + 'spike_abs_C.npy')
spike_abs_D = np.load(path3 + 'spike_abs_D.npy')

spike_B_lower_noise = np.load(path3 + 'spike_B_lower_noise.npy')
spike_C_lower_noise = np.load(path3 + 'spike_C_lower_noise.npy')
spike_abs_B_lower_noise = np.load(path3 + 'spike_abs_B_lower_noise.npy')
spike_abs_C_lower_noise = np.load(path3 + 'spike_abs_C_lower_noise.npy')

BCD_distance = np.load(path3 + 'BCD_distance.npy')
BCD_abs_distance = np.load(path3 + 'BCD_abs_distance.npy')


# %%
print('preprocessing test data...')
spike_B_sum = spike_B.sum(axis=1)
spike_C_sum = spike_C.sum(axis=1)
spike_D_sum = spike_D.sum(axis=1)

spike_B_lower_noise_sum = spike_B_lower_noise.sum(axis=1)
spike_C_lower_noise_sum = spike_C_lower_noise.sum(axis=1)

spike_B_sg_B_sum = np.array([spike_B[i]-sg_B[i]
                            for i in range(len(spike_B))]).sum(axis=1)
spike_C_sg_D_sum = np.array([spike_C[i]-sg_D[i]
                             for i in range(len(spike_C))]).sum(axis=1)
spike_D_sg_F_sum = np.array([spike_D[i]-sg_F[i]
                             for i in range(len(spike_D))]).sum(axis=1)

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

integrated_spike_B_sg_B_sum = [sum(abs(spike_B_sg_B_sum[:i]))
                      for i in range(1, len(spike_B_sg_B_sum)+1)]
df2['integrated_spike_B_sg_B_sum'] = pd.Series(integrated_spike_B_sg_B_sum)

integrated_spike_C_sg_D_sum = [sum(abs(spike_C_sg_D_sum[:i]))
                               for i in range(1, len(spike_C_sg_D_sum)+1)]
df2['integrated_spike_C_sg_D_sum'] = pd.Series(integrated_spike_C_sg_D_sum)

integrated_spike_D_sg_F_sum = [sum(abs(spike_D_sg_F_sum[:i]))
                               for i in range(1, len(spike_D_sg_F_sum)+1)]
df2['integrated_spike_D_sg_F_sum'] = pd.Series(integrated_spike_D_sg_F_sum)

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

Input = Input_transformer.transform(Input)

print('Input layer 0: ', Input[0, :10])


# %%
total_predict = model.predict(Input)


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



