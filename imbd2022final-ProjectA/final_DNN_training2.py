from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras import backend as K

import numpy as np
import pandas as pd
import tensorflow as tf
import time

from train_processing import feature, corr, aggregate_previous
from test_processing import df


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#threshold = 0.0
random_state = 42
learning_rate = 0.003
batch_size = 40
epochs = 500
dropout = 0.2

Model = feature

print('Model shape: ', Model.shape)
Model = Model.values

Input = Model[:, :-1]
Output = Model[:, -1].reshape((-1, 1))

Input_transformer = StandardScaler()    # Standard
Output_transformer = MaxAbsScaler()


Input = Input_transformer.fit_transform(Input)
#Output = Output_transformer.fit_transform(Output)

input_train, input_test, output_train, output_test = train_test_split(
    Input, Output, test_size=0.1, random_state=random_state)


###########################################################################################################################
input_train = tf.convert_to_tensor(input_train)
input_test = tf.convert_to_tensor(input_test)
output_train = tf.convert_to_tensor(output_train)
output_test = tf.convert_to_tensor(output_test)

print(input_train.shape)
print(input_test.shape)
print(output_train.shape)
print(output_test.shape)


model = Sequential()
model.add(Dense(512, activation='relu', input_dim=input_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=tf.optimizers.Adam(
    learning_rate=learning_rate), loss=rmse)

start = time.time()
history = model.fit(input_train, output_train, validation_data=(input_train, output_train),
                    batch_size=batch_size, epochs=epochs, verbose=1)
end = time.time()


###################################################################################################################

def numpy_rmse(true, predict):
    return pow(np.mean(pow(true - predict, 2)), 0.5)

##################################################################################################################


# fine tune
################################################################################################################
print('\n\n\n')
print('Fine Tune...', end='\n\n\n')

model.compile(optimizer=tf.optimizers.RMSprop(
    learning_rate=learning_rate/10), loss=rmse)

start = time.time()

history = model.fit(input_train, output_train, validation_data=(input_test, output_test),
                    batch_size=batch_size, epochs=int(epochs/2), verbose=1)
end = time.time()

##################################################################################################################


print('\n\n')
#print('threshold:\t', threshold)
print('random state:\t', random_state)
print('learning rate:\t', learning_rate)
print('batch size:\t', batch_size)
print('epochs:\t\t', epochs)
print('dropout:\t', dropout)


test_predict = model.predict(input_test)
test_predict_true = output_test.numpy()

#test_predict = Output_transformer.inverse_transform(test_predict)
#test_predict_true = Output_transformer.inverse_transform(test_predict_true)

test_predict = np.round(test_predict, 0)
test_RMSE = numpy_rmse(test_predict_true, test_predict)
print('Test:')
print('Test RMSE:\t', test_RMSE)
print('Test true:\t', test_predict_true[:6].T)
print('Test predict:\t', test_predict[:6].T)
print()


############################################################################################

train_predict = model.predict(input_train)
train_predict_true = output_train.numpy()

#train_predict = Output_transformer.inverse_transform(train_predict)
#train_predict_true = Output_transformer.inverse_transform(train_predict_true)

train_predict = np.round(train_predict, 0)
train_RMSE = numpy_rmse(train_predict_true, train_predict)
print('Train:')
print('Train RMSE:\t', train_RMSE)
print('Train true:\t', train_predict_true[:6].T)
print('Train predict:\t', train_predict[:6].T)
print()

##################################################################################################


total = model.predict(Input)
Output = Output

#total = Output_transformer.inverse_transform(total)
#Output = Output_transformer.inverse_transform(Output)

total = np.round(total, 0)
total_RMSE = numpy_rmse(Output, total)
print('Total: ')
print('Total RMSE:\t', total_RMSE)
print('Total true:\t', total[:6].T)
print('Total predict:\t', Output[:6].T)
print()
