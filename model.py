# %%
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split


# %%
# Define feature index and load .npy
Sg_Feature_Index = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
Spike_Feature_Index = ['B', 'C', 'D']

sg = {}
spike = {}

for i in Sg_Feature_Index:
    sg['sg_' + str(i)] = np.load('sg_' + str(i) + '.npy')
    
for i in Spike_Feature_Index:
    spike['spike_' + str(i)] = np.load('spike_' + str(i) + '.npy')
    
sg_feture = list(sg.keys())
spike_feture = list(spike.keys())


# %%
# Define Output and get difference between two layers
Output = pd.read_csv('train1/00_Wear_data.csv').loc[:, 'MaxWear'].values
a = [Output[i] - Output[i-1] for i in range(1, Output.shape[0])]
a = np.insert(a, 0, 0)
Output = np.reshape(a, (-1, 1))

# %%
Scaler = {}

for i in sg_feture:
    if i == 'sg_I':
        Scaler[i] = MaxAbsScaler()
    else:
        Scaler[i] = StandardScaler()
    sg[i] = Scaler[i].fit_transform(sg[i])
for i in spike_feture:
    Scaler[i] = StandardScaler()
    spike[i] = Scaler[i].fit_transform(spike[i])

Scaler_output = StandardScaler()
Output = Scaler_output.fit_transform(Output)

# %%
feature_list = []

for i in sg_feture:
    list.append(feature_list, sg[i])

for i in spike_feture:
    list.append(feature_list, spike[i])

feature_Input = np.array(feature_list)
print(feature_Input.shape)

feature_Input = feature_Input.transpose(2, 0, 1)
print(feature_Input.shape)


# %%
input_train, input_test, output_train, output_test = train_test_split(
    feature_Input, Output, test_size=0.1, random_state=42)

print(input_train.shape)
print(input_test.shape)
print(output_train.shape)
print(output_test.shape)


# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


input_train = tf.convert_to_tensor(input_train)
input_test = tf.convert_to_tensor(input_test)
output_train = tf.convert_to_tensor(output_train)
output_test = tf.convert_to_tensor(output_test)


model = Sequential()
model.add(Conv1D(512, 1, activation='relu',
          input_shape=input_train.shape[1:]))
model.add(BatchNormalization())
model.add(Conv1D(512, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(512, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# model.add(Conv1D(512, 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv1D(512, 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv1D(512, 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

model.add(Conv1D(512, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(512, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(512, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# model.add(Conv1D(256, 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv1D(256, 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv1D(256, 1, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))

model.add(Conv1D(256, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(128, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(128, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(128, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.summary()


# %%
model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001), loss=rmse)
history = model.fit(input_train, output_train,
                    validation_data=(input_test, output_test), batch_size=32, epochs=70, verbose=1)


# %%
def numpy_rmse(actual, predict, Scaler_output):
    a = Scaler_output.inverse_transform(actual)
    b = Scaler_output.inverse_transform(predict)
    return pow(np.mean(pow(a - b, 2)), 0.5)
    # return pow(np.mean(pow(actual - predict, 2)), 0.5)


test_predict = model.predict(input_test)
test_predict_actual = output_test.numpy()
test_RMSE = numpy_rmse(test_predict_actual, test_predict, Scaler_output)
print('Test: ')
print('Test RMSE:\t', test_RMSE)
print()


train_predict = model.predict(input_train)
train_predict_actual = output_train.numpy()
train_RMSE = numpy_rmse(train_predict_actual, train_predict, Scaler_output)
print('Train: ')
print('Train RMSE:\t', train_RMSE)


# %%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')


