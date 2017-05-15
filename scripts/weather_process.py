import pandas as pd

PATH = 'scripts/'

weather_file = 'weather (table 7)_training_update.csv'  
data = pd.read_csv(PATH + weather_file)
data = data.fillna('missing')

for i in range(data.shape[0]):
    if any(['missing' in data.loc[i, :].values,
        data.loc[i, 'hour'] not in range(25),
        data.loc[i, 'pressure'] > 1500,
        data.loc[i, 'wind_direction'] < 0 or data.loc[i, 'wind_direction'] > 360,
        data.loc[i, 'wind_speed'] > 10,
        data.loc[i, 'precipitation'] > 10]):
        print('已删除存在异常值 %s 行数据'%i)
        data.drop([i], inplace = True)
        
data.to_csv('weather.csv')