import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)

anomaly_train1 = pd.read_csv('train1/anomaly_train1.csv').sort_values(by=['oven_id', 'layer_id', 'anomaly_accumulation_hour'])
#anomaly_train1 = pd.read_csv('train1/anomaly_train1.csv')

anomaly_train1.loc[:, 'lamp_id'] = pd.Series([list(map(int, string.split('_'))) for string in anomaly_train1.loc[:, 'lamp_id']])
accumulation_hour1 = pd.read_csv('train1/accumulation_hour1.csv')


#a = accumulation_hour1[(accumulation_hour1['oven_id'] == '1B0') & (accumulation_hour1['layer_id'] == 3)].values[0][3]

hour = [accumulation_hour1[(accumulation_hour1['oven_id'] == anomaly_train1.loc[i, 'oven_id']) & (accumulation_hour1['layer_id'] == anomaly_train1.loc[i, 'layer_id'])].values[0][3]  
        for i in range(anomaly_train1.shape[0])]
df = anomaly_train1

df.insert(5, '5/4 oven hour', pd.Series(hour))
df.insert(6, 'difference', df['5/4 oven hour'] - df['anomaly_accumulation_hour'])
df.insert(7, '    check', pd.Series(['N' if df.loc[i, 'difference'] < 0 else ' '  for i in range(anomaly_train1.shape[0])]))

#---separate A and B
count = []
for datanumber in range(anomaly_train1.shape[0]):
AB = [0, 0]
for lamp_id in anomaly_train1.loc[datanumber, 'lamp_id']:
if lamp_id not in [1, 2, 60, 61, 62, 63, 121, 122]:
AB[0] += 1
else:
AB[1] += 1
count.append(AB)

df.insert(8, 'A count', pd.Series([i[0] for i in count]))# A count: other lamp
df.insert(9, 'B count', pd.Series([i[1] for i in count]))# B count: lamp



#----separate 1-61 and 62-122
"""
section = []

for datanumber in range(anomaly_train1.shape[0]):
AB = [0, 0]
for lamp_id in anomaly_train1.loc[datanumber, 'lamp_id']:
if lamp_id <= 61:
AB[0] += 1
else:
AB[1] += 1
section.append(AB)


df.insert(10, 'total count', df['A count'] + df['B count'])
df.insert(11, '1-61', pd.Series([i[0] for i in section]))
df.insert(12, '62-122', pd.Series([i[1] for i in section]))



df = df.sort_values(by=['oven_id', 'layer_id', 'anomaly_accumulation_hour'])
df = df.drop(columns=['anomaly_total_number'])
"""

df = df.drop(columns=['A count', 'B count'])
print(df)
