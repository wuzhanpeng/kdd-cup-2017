# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd
import numpy as np

routes = ['A-2','A-3','B-1','B-3','C-1','C-3']
weekdays = [1,2,3,4,5,6,0]
ampms = {
	'am': ['6-0','6-1','6-2','7-0','7-1','7-2','8-0','8-1','8-2','9-0','9-1','9-2'],
	'pm': ['15-0','15-1','15-2','16-0','16-1','16-2','17-0','17-1','17-2','18-0','18-1','18-2']}

traindatapath = '../data/training_20min_avg_travel_time.csv'
testdatapath = '../data/test1_20min_avg_travel_time.csv'

def loadfromcsv(path):
	agg_csv_file = open(path)
	columns = agg_csv_file.readline()

	rawdata = []
	for line in agg_csv_file:
		fields = line.strip("\n").replace('"', '').split(',')
		fields[2] = datetime.strptime(fields[2][1:], "%Y-%m-%d %H:%M:%S")
		fields[3] = datetime.strptime(fields[3][:-1], "%Y-%m-%d %H:%M:%S")
		fields[4] = float(fields[4])
		rawdata.append(fields)

	# drop redundancy time-interval
	rawdata = filter(lambda fields: (
		fields[2].day == fields[3].day and (
		fields[2].hour >= 6 and 
		fields[2].hour < 10 or 
		fields[2].hour >= 15 and 
		fields[2].hour < 19)), rawdata)

	# re-construct data
	records = []
	for fields in rawdata:
		records.append([
			fields[2],
			fields[2].date(),
			str(fields[0]+'-'+fields[1]),
			fields[2].weekday(),
			str(fields[2].hour)+'-'+str(fields[2].minute/20),
			'am' if fields[2].hour < 12 else 'pm',
			fields[4]])

	# print(features, label)
	return pd.DataFrame(records, columns = ['datetime','date','route','weekday','time-interval','am-pm','avg-time'])

# print (loadfromcsv(traindatapath))
# print (loadfromcsv(testdatapath))

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]

def best_diff(df, maxdiff = 8):
    p_set = {}
    for i in range(0, maxdiff):
        temp = df.copy() #每次循环前，重置
        if i == 0:
            temp['diff'] = temp[temp.columns[0]]
        else:
            temp['diff'] = temp[temp.columns[0]].diff(i)
            temp = temp.drop(temp.iloc[:i].index) #差分后，前几行的数据会变成nan，所以删掉
        pvalue = test_stationarity(temp['diff'])
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set, orient="index")
        p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df['p_value'][i]<0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff