import pandas as pd
import numpy as np

accumulation_hour = pd.read_csv("train1/accumulation_hour1.csv")
anomaly_train = pd.read_csv('train1/anomaly_train1.csv')
cooler = pd.read_csv("train1/cooler.csv")
power = pd.read_csv("train1/power.csv")


accumulation_hour = accumulation_hour.rename(columns={'accumulation_hour': 'anomaly_accumulation_hour'})


#anomaly_train1 = pd.read_csv('train1/anomaly_train1.csv').sort_values(by=['oven_id', 'layer_id', 'anomaly_accumulation_hour'])
anomaly_train.loc[:, 'lamp_id'] = pd.Series([list(map(int, string.split('_'))) for string in anomaly_train.loc[:, 'lamp_id']])

accumulation_hour['lamp_id'] = pd.Series([[] for i in range(accumulation_hour.shape[0])])



anomaly_train = pd.concat([anomaly_train, accumulation_hour],
                           ignore_index=True).sort_values(by=['oven_id', 'layer_id', 'anomaly_accumulation_hour'])


# change 'accumulation_hour' column's item from string to list (int)
power.loc[:, "accumulation_hour"] = pd.Series([list(map(int, string.split("-")))
                                                     for string in power.loc[:, "accumulation_hour"]])



#######################################################################################################

hour = [accumulation_hour[(accumulation_hour['oven_id'] == anomaly_train.loc[i, 'oven_id']) & (accumulation_hour['layer_id'] == anomaly_train.loc[i, 'layer_id'])].values[0][3]
        for i in range(anomaly_train.shape[0])]

anomaly_train['5/4 oven hour'] = pd.Series(hour)
anomaly_train['    check'] = pd.Series(['N' if anomaly_train.loc[i, 'anomaly_accumulation_hour'] > anomaly_train.loc[i, '5/4 oven hour'] else ' ' for i in range(anomaly_train.shape[0])])
anomaly_train = anomaly_train[anomaly_train['    check'] != 'N']

anomaly_train = anomaly_train.drop(columns=['5/4 oven hour', '    check'])
anomaly_train = anomaly_train.reset_index(drop=True)


count = []
for datanumber in range(anomaly_train.shape[0]):
        AB = [0, 0]
        for lamp_id in anomaly_train.loc[datanumber, 'lamp_id']:
                if lamp_id not in [1, 2, 60, 61, 62, 63, 121, 122]:
                        AB[0] += 1
                else:
                        AB[1] += 1
        count.append(AB)
anomaly_train['A count'] = pd.Series([i[0] for i in count])
anomaly_train['B count'] = pd.Series([i[1] for i in count])
anomaly_train['total count'] = anomaly_train['A count'] + anomaly_train['B count']
anomaly_train = anomaly_train.drop(columns=['lamp_id', 'anomaly_total_number', 'A count', 'B count'])


#################################################################################

aggregate_previous = []# aggregate number at 5/4 or 6/2
aggregate = []
current_oven = '1B0'
current_layer = 1
count = 0
index = 0

for oven in anomaly_train['oven_id']:
    for layer in anomaly_train['layer_id']:
        if oven == current_oven and layer == current_layer and index < anomaly_train.shape[0]:
            count += anomaly_train.loc[index, 'total count']
            aggregate.append(count)
        elif index < anomaly_train.shape[0]:
            current_oven = oven
            current_layer = layer
            count = 0
            count += anomaly_train.loc[index, 'total count']
            aggregate.append(count)
            aggregate_previous.append(aggregate[-2])
        index += 1
aggregate_previous.append(aggregate[-1])


anomaly_train['aggregate anomalies'] = pd.Series(aggregate)
print()
anomaly_train = anomaly_train.drop(columns=['date', 'total count'])

print('before drop duplicates: ', anomaly_train.shape)
anomaly_train = anomaly_train.drop_duplicates()
anomaly_train = anomaly_train.reset_index(drop=True)
print('after drop duplicates: ', anomaly_train.shape)


df = anomaly_train
pd.set_option('display.max_rows', None)
print(df)
print()
print(len(aggregate_previous))
print(aggregate_previous)
#########################################################################################################

#----power----

power_AB = []
for hour in df.loc[:, 'anomaly_accumulation_hour']:
    index, _AB = 0, [0, 0]
    for i in power.loc[:, 'accumulation_hour']:
        if hour > i[1]:
            index += 1
            continue
        else:
            _AB[0] = power.loc[index, 'power_setup(other_lamp)']
            _AB[1] = power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)']
    power_AB.append(_AB)

df['power_A'] = pd.Series([i[0] for i in power_AB])
df['power_B'] = pd.Series([i[1] for i in power_AB])
df['power_A_2'] = np.power(df['power_A'], 2)
df['power_B_2'] = np.power(df['power_B'], 2)
df['power_A_3'] = np.power(df['power_A'], 3)
df['power_B_3'] = np.power(df['power_B'], 3)
df['power_A_4'] = np.power(df['power_A'], 4)
df['power_B_4'] = np.power(df['power_B'], 4)




df['power_A_2 * hour_3'] = np.power(df['power_A'], 2) * np.power(df['anomaly_accumulation_hour'], 3)
df['power_B_2 * hour_3'] = np.power(df['power_B'], 2) * np.power(df['anomaly_accumulation_hour'], 3)

df['power_A_6'] = np.power(df['power_A'], 6)
df['power_B_6'] = np.power(df['power_B'], 6)



#########################################################################################################

#----energy (power * time)----

energy = []
for hour in df.loc[:, 'anomaly_accumulation_hour']:
    index, powers = 0, [0, 0]
    for i in power.loc[:, 'accumulation_hour']:
        if index == 0:
            if hour >= i[1]:
                powers[0] += (i[1] - i[0]) * power.loc[index, 'power_setup(other_lamp)']
                powers[1] += (i[1] - i[0]) * power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)']
            else:
                powers[0] = hour * power.loc[index, 'power_setup(other_lamp)']
                powers[1] = hour * power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)']
                break
        elif hour >= i[1]:
            powers[0] += (i[1] - i[0] + 1) * power.loc[index, 'power_setup(other_lamp)']
            powers[1] += (i[1] - i[0] + 1) * power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)']
        else:
            powers[0] += (hour - i[0] + 1) * power.loc[index, 'power_setup(other_lamp)']
            powers[1] += (hour - i[0] + 1) * power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)']
            break
        index += 1
    energy.append(powers)

df['energy_A'] = pd.Series([i[0] for i in energy])
df['energy_B'] = pd.Series([i[1] for i in energy])



#----energy (power^10 * time)----
energy2 = []
for hour in df.loc[:, 'anomaly_accumulation_hour']:
    index, powers = 0, [0, 0]
    for i in power.loc[:, 'accumulation_hour']:
        if index == 0:
            if hour >= i[1]:
                powers[0] += (i[1] - i[0]) * np.power(power.loc[index, 'power_setup(other_lamp)'],10)
                powers[1] += (i[1] - i[0]) * np.power(power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)'], 10)
            else:
                powers[0] = hour * np.power(power.loc[index, 'power_setup(other_lamp)'],10)
                powers[1] = hour * np.power(power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)'], 10)
                break
        elif hour >= i[1]:
            powers[0] += (i[1] - i[0] + 1) * np.power(power.loc[index, 'power_setup(other_lamp)'],10)
            powers[1] += (i[1] - i[0] + 1) * np.power(power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)'], 10)
        else:
            powers[0] += (hour - i[0] + 1) * np.power(power.loc[index, 'power_setup(other_lamp)'],10)
            powers[1] += (hour - i[0] + 1) * np.power(power.loc[index, 'power_setup(lamp_1_2_60_61_62_63_121_122)'], 10)
            break
        index += 1
    energy2.append(powers)

df['energy_A_10'] = pd.Series([i[0] for i in energy2])
df['energy_B_10'] = pd.Series([i[1] for i in energy2])

#########################################################################################################

cooling_system = pd.DataFrame()


cooling_system['upper_water_volume'] = pd.Series([i for oven in cooler.columns[1: ] for i in cooler.loc[0:18, oven]])
cooling_system['lower_water_volume'] = pd.Series([i for oven in cooler.columns[1: ] for i in cooler.loc[1:19, oven]])

# water temperature
cooling_system['upper_water_in_temperature'] = pd.Series([i for oven in cooler.columns[1:]
                                                for i in [cooler.loc[20, oven]]*10 + [cooler.loc[22, oven]]*9])
cooling_system['lower_water_in_temperature'] = pd.Series([i for oven in cooler.columns[1:]
                                                for i in [cooler.loc[20, oven]]*9 + [cooler.loc[22, oven]]*10] )
cooling_system['upper_water_out_temperature'] = pd.Series([i for oven in cooler.columns[1: ]
                                                 for i in [cooler.loc[21, oven]]*10 + [cooler.loc[23, oven]]*9])
cooling_system['lower_water_out_temperature'] = pd.Series([i for oven in cooler.columns[1: ]
                                                 for i in [cooler.loc[21, oven]]*9 + [cooler.loc[23, oven]]*10])


# S A/B temperature
cooling_system['S_A_temperature'] = pd.Series([i for oven in cooler.columns[1:] for i in cooler.loc[np.arange(24, 61, 2), oven]])
cooling_system['S_B_temperature'] = pd.Series([i for oven in cooler.columns[1:] for i in cooler.loc[np.arange(25, 62, 2), oven]])


#########################################################################################################

oven_id = ['1B0', '1C0', '1D0', '1E0', '1G0', 
           '2B0', '2C0', '2D0', '2E0', '2G0']
layer_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

upper_water_volume = []
lower_water_volume = []
upper_water_in_temperature = []
lower_water_in_temperature = []
upper_water_out_temperature = []
lower_water_out_temperature = []
S_A_temperature = []
S_B_temperature = []


for i in range(df.shape[0]):
    position = position = oven_id.index(df.loc[i, 'oven_id'])*19 + layer_id.index(df.loc[i, 'layer_id'])
    upper_water_volume.append(cooling_system.loc[position, 'upper_water_volume'])
    lower_water_volume.append(cooling_system.loc[position, 'lower_water_volume'])
    upper_water_in_temperature.append(cooling_system.loc[position, 'upper_water_in_temperature'])
    lower_water_in_temperature.append(cooling_system.loc[position, 'lower_water_in_temperature'])
    upper_water_out_temperature.append(cooling_system.loc[position, 'upper_water_out_temperature'])
    lower_water_out_temperature.append(cooling_system.loc[position, 'lower_water_out_temperature'])
    S_A_temperature.append(cooling_system.loc[position, 'S_A_temperature'])
    S_B_temperature.append(cooling_system.loc[position, 'S_B_temperature'])

df['upper_water_volume'] = pd.Series(upper_water_volume)
df['lower_water_volume'] = pd.Series(lower_water_volume)
df['upper_water_in_temperature'] = pd.Series(upper_water_in_temperature)
df['lower_water_in_temperature'] = pd.Series(lower_water_in_temperature)
df['upper_water_out_temperature'] = pd.Series(upper_water_out_temperature)
df['lower_water_out_temperature'] = pd.Series(lower_water_out_temperature)
df['S_A_temperature'] = pd.Series(S_A_temperature)
df['S_B_temperature'] = pd.Series(S_B_temperature)

#########################################################################################################

df['anomaly_accumulation_hour^2'] = np.power(df['anomaly_accumulation_hour'], 2)
df['anomaly_accumulation_hour^3'] = np.power(df['anomaly_accumulation_hour'], 3)
df['anomaly_accumulation_hour^4'] = np.power(df['anomaly_accumulation_hour'], 4)

df['upper_water_volume'] = np.log1p(df['upper_water_volume'])
df['lower_water_volume'] = np.log1p(df['lower_water_volume'])
df['upper_water_volume'] = np.power(df['upper_water_volume'], 2)
df['lower_water_volume'] = np.power(df['lower_water_volume'], 2)

df['water_volume'] = df['upper_water_volume'] + df['lower_water_volume']
df['water_volume^2'] = np.power(df['water_volume'], 2)

df['average_water_in_temperature'] = (df['upper_water_in_temperature'] + df['lower_water_in_temperature']) / 2
df['average_water_out_temperature'] = (df['upper_water_out_temperature'] + df['lower_water_in_temperature']) / 2
df['average_water_in_out_dif'] = df['average_water_out_temperature'] - df['average_water_in_temperature']

# water temperatrue difference
df['upper_water_temperature_dif'] = df['upper_water_out_temperature'] - df['upper_water_in_temperature']
df['lower_water_temperature_dif'] = df['lower_water_out_temperature'] - df['lower_water_in_temperature']
df['up_w_t_d ^2'] = np.power(df['upper_water_temperature_dif'], 2)
df['lo_w_t_d ^2'] = np.power(df['lower_water_temperature_dif'], 2)


# water heat absorption
df['upper_water_heat_absorption'] = df['upper_water_volume'] * df['upper_water_temperature_dif']
df['lower_water_heat_absorption'] = df['lower_water_volume'] * df['lower_water_temperature_dif']
df['up_w_v * up_w_t_d ^2'] = df['upper_water_volume'] * np.power(df['upper_water_temperature_dif'], 2)
df['lo_w_v * lo_w_t_d ^2'] = df['lower_water_volume'] * np.power(df['lower_water_temperature_dif'], 2)
df['up_w_v * up_w_t_d ^3'] = df['upper_water_volume'] * np.power(df['upper_water_temperature_dif'], 3)
df['up_w_v * up_w_t_d ^4'] = df['upper_water_volume'] * np.power(df['upper_water_temperature_dif'], 4)

df['heat_absorption'] = df['upper_water_heat_absorption'] + df['lower_water_heat_absorption']
df['energy_A / absorption'] = df['energy_A'] / df['heat_absorption']
df['energy_B / absorption'] = df['energy_B'] / df['heat_absorption']

#----energy per contact volume
df['energy_per_volume_A'] = df['energy_A'] / df['water_volume']
df['energy_per_volume_B'] = df['energy_B'] / df['water_volume']

df['A_upper_energy_take_away'] = df['energy_A'] / df['upper_water_temperature_dif']
df['A_lower_energy_take_away'] = df['energy_A'] / df['lower_water_temperature_dif']
df['B_upper_energy_take_away'] = df['energy_B'] / df['upper_water_temperature_dif']
df['B_lower_energy_take_away'] = df['energy_B'] / df['lower_water_temperature_dif']

df['energy_A_upper/heat_absorption'] = df['energy_A'] / df['upper_water_heat_absorption']
df['energy_A_lower/heat_absorption'] = df['energy_A'] / df['lower_water_heat_absorption']
df['energy_B_upper/heat_absorption'] = df['energy_B'] / df['upper_water_heat_absorption']
df['energy_B_lower/heat_absorption'] = df['energy_B'] / df['lower_water_heat_absorption']
df['energy_A / heat_absorption'] = df['energy_A'] / df['heat_absorption']
df['energy_B / heat_absorption'] = df['energy_B'] / df['heat_absorption']

# S A/B temperature
df['S_A_temperature - lower_water_out_temperature'] = df['S_A_temperature'] - df['lower_water_out_temperature']
df['S_A_temperature - lower_water_in_temperature'] = df['S_A_temperature'] - df['lower_water_in_temperature']
df['S_B_temperature - lower_water_out_temperature'] = df['S_B_temperature'] - df['lower_water_out_temperature']
df['S_B_temperature - lower_water_in_temperature'] = df['S_B_temperature'] - df['lower_water_in_temperature']

df['power_A - S_A_temperature'] = df['power_A'] - df['S_A_temperature']
df['power_B - S_A_temperature'] = df['power_B'] - df['S_A_temperature']
df['power_A - S_B_temperature'] = df['power_A'] - df['S_B_temperature']
df['power_B - S_B_temperature'] = df['power_B'] - df['S_B_temperature']

df['(power_A - S_A_temperature)^2'] = np.power(df['power_A - S_A_temperature'], 2)
df['(power_B - S_A_temperature)^2'] = np.power(df['power_B - S_A_temperature'], 2)
df['(power_A - S_B_temperature)^2'] = np.power(df['power_A - S_B_temperature'], 2)
df['(power_B - S_B_temperature)^2'] = np.power(df['power_B - S_B_temperature'], 2)

df['power_A - (S_A_temp - lower_water_out_temp)'] = df['power_A'] - df['S_A_temperature - lower_water_out_temperature']
df['power_A - (S_A_temp - lower_water_in_temp)'] = df['power_A'] - df['S_A_temperature - lower_water_in_temperature']
df['power_A - (S_B_temp - lower_water_out_temp)'] = df['power_A'] - df['S_B_temperature - lower_water_out_temperature']
df['power_A - (S_B_temp - lower_water_in_temp)'] = df['power_A'] - df['S_B_temperature - lower_water_in_temperature']
df['power_B - (S_A_temp - lower_water_out_temp)'] = df['power_B'] - df['S_A_temperature - lower_water_out_temperature']
df['power_B - (S_A_temp - lower_water_in_temp)'] = df['power_B'] - df['S_A_temperature - lower_water_in_temperature']
df['power_B - (S_B_temp - lower_water_out_temp)'] = df['power_B'] - df['S_B_temperature - lower_water_out_temperature']
df['power_B - (S_B_temp - lower_water_in_temp)'] = df['power_B'] - df['S_B_temperature - lower_water_in_temperature']

df['(S_A_temp - lower_water_out_temp) / power_A'] = df['S_A_temperature - lower_water_out_temperature'] / df['power_A']
df['(S_A_temp - lower_water_in_temp) / power_A'] =  df['S_A_temperature - lower_water_in_temperature'] / df['power_A']
df['(S_B_temp - lower_water_out_temp) / power_A'] = df['S_B_temperature - lower_water_out_temperature'] / df['power_A']
df['(S_B_temp - lower_water_in_temp) / power_A'] = df['S_B_temperature - lower_water_in_temperature'] / df['power_A']
df['(S_A_temp - lower_water_out_temp) / power_B'] = df['S_A_temperature - lower_water_out_temperature'] / df['power_B']
df['(S_A_temp - lower_water_in_temp) / power_B'] = df['S_A_temperature - lower_water_in_temperature'] / df['power_B']
df['(S_B_temp - lower_water_out_temp) / power_B'] = df['S_B_temperature - lower_water_out_temperature'] / df['power_B']
df['(S_B_temp - lower_water_in_temp) / power_B'] = df['S_B_temperature - lower_water_in_temperature'] / df['power_B']

df['upper_total_heat_absorption'] = df['upper_water_heat_absorption'] * df['anomaly_accumulation_hour']
df['lower_total_heat_absorption'] = df['lower_water_heat_absorption'] * df['anomaly_accumulation_hour']

df['upper_water_temperature_dif * hour'] = df['upper_water_temperature_dif'] * df['anomaly_accumulation_hour']
df['lower_water_temperature_dif * hour'] = df['lower_water_temperature_dif'] * df['anomaly_accumulation_hour']

df['hour / volume'] = df['anomaly_accumulation_hour'] / df['water_volume']
df['hour * volume'] = df['anomaly_accumulation_hour'] * df['water_volume']

df['upper_water_temperature_dif*hour^2'] = df['upper_water_temperature_dif'] * np.power(df['anomaly_accumulation_hour'], 2)
df['upper_water_temperature_dif*hour^3'] = df['upper_water_temperature_dif'] * np.power(df['anomaly_accumulation_hour'], 3)
df['upper_water_temperature_dif*hour^4'] = df['upper_water_temperature_dif'] * np.power(df['anomaly_accumulation_hour'], 4)

#########################################################################################################


print(df.shape)

cols = df.columns.tolist()
col = cols[:3] + cols[4:]
col.append(cols[3])
df = df[col]



from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
feature1 = df.iloc[:, 2:]

#pd.set_option('display.max_rows', None)
corr1 = feature1.corr()

111052@c1team31-9b7c9586c-kppmn:/home/projectA/Spare$ 