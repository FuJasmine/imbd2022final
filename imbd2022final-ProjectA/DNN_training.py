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
random_state = 43
learning_rate = 0.003# 0.003 0.005
batch_size = 40
epochs = 500
dropout = 0.2


Model = feature


Model = shuffle(Model, random_state=random_state)
Test = df


print('Model shape: ', Model.shape)
print('Test shape: ', Test.shape)

Model = Model.values
Input = Model[:, :-1]
Output = Model[:, -1].reshape((-1, 1))

Input_transformer = StandardScaler()    # Standard
Output_transformer = MaxAbsScaler()


Input = Input_transformer.fit_transform(Input)
Test = Input_transformer.transform(Test)

#Output = Output_transformer.fit_transform(Output)



input_train, input_test, output_train, output_test = Input, Input, Output, Output



###########################################################################################################################
input_train = tf.convert_to_tensor(input_train)
input_test = tf.convert_to_tensor(input_test)
output_train = tf.convert_to_tensor(output_train)
output_test = tf.convert_to_tensor(output_test)
Test = tf.convert_to_tensor(Test)

print(input_train.shape)
print(input_test.shape)
print(output_train.shape)
print(output_test.shape)



model = Sequential()
model.add(Dense(512, activation='relu', input_dim = input_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=rmse)

start = time.time()
history = model.fit(input_train, output_train, validation_data=(input_train, output_train),
                    batch_size=batch_size, epochs=epochs, verbose=1)
end = time.time()




#fine tune
################################################################################################################
print('\n\n\n')
print('Fine Tune...', end='\n\n\n')

model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate/10), loss=rmse)

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



###################################################################################################################

def numpy_rmse(true, predict):
        return pow(np.mean(pow(true - predict, 2)), 0.5)

##################################################################################################################



total = model.predict(Input)
Output_final_tune = Output

total = np.round(total, 0)
total_RMSE = numpy_rmse(Output_final_tune, total)
print('Total: ')
print('Total RMSE: \t', total_RMSE)
print()


#######################################################################################

answer = model.predict(Test)
#answer = Output_transformer.inverse_transform(answer)

answer = np.round(answer, 0)

answer = np.reshape(answer, (-1,))


#final_answer = answer - aggregate_previous
#print(final_answer)



"""
for i in range(final_answer.shape[0]):
if final_answer[i] < 0:
final_answer[i] = 0
"""
#print(final_answer)

answer_sheet = pd.read_csv('projectA_template.csv')
#answer_sheet['anomaly_total_number'] = pd.Series(final_answer)
answer_sheet['anomaly_total_number'] = pd.Series(answer)
answer_sheet.to_csv('111052_projectA_ans.csv', mode='w', index=False)











