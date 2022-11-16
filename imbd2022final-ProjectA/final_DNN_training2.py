y
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

#from train1_processing import feature1, corr1
#from train2_processing import feature2, corr2

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

#Model1 = feature1.loc[:, corr1[abs(corr1.iloc[-1]) > threshold].index]
#Model2 = feature2.loc[:, corr2[abs(corr2.iloc[-1]) > threshold].index]

#Model1 = feature1
#Model2 = feature2





#Model = pd.concat([Model1, Model2, df])
#Model = Model1
Model = feature


Model = shuffle(Model, random_state=random_state)

#Test = df
Test = Model.iloc[:, :-1]



print('Model shape: ', Model.shape)
print('Test shape: ', Test.shape)




#print(Model.columns)
Model = Model.values

Input = Model[:, :-1]
Output = Model[:, -1].reshape((-1, 1))





Input_transformer = StandardScaler()    # Standard
Output_transformer = MaxAbsScaler()


Input = Input_transformer.fit_transform(Input)
#Output = Output_transformer.fit_transform(Output)
Test = Input_transformer.transform(Test)



#input_train, input_test, output_train, output_test = train_test_split(Input, Output, test_size=0.0, random_state=random_state)
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


###################################################################################################################

def numpy_rmse(true, predict):
        return pow(np.mean(pow(true - predict, 2)), 0.5)

##################################################################################################################


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


answer = model.predict(Test)
Output_final_tune = Output

#answer = Output_transformer.inverse_transform(answer)
#Output_final_tune = Output_transformer.inverse_transform(Output)

answer = np.round(answer, 0)
total_RMSE = numpy_rmse(Output_final_tune, answer)
print('Total: ')
print('Total RMSE: \t', total_RMSE)
print()



"""

final_answer = np.round(answer, 0)
answer_sheet = pd.read_csv('answer.csv')
answer_sheet['anomaly_total_number'] = pd.Series(final_answer)
answer_sheet = answer_sheet.reset_index(drop=True)
answer_sheet.to_csv('answer.csv', mode='w', index=False)

"""










