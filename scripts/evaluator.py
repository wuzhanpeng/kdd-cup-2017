# -*- coding: utf-8 -*-

from commons import *
import numpy as np
import matplotlib.pyplot as plt

lassodata = loadfromcsv(lasso)
arimadata = loadfromcsv(arima)
averagedata = loadfromcsv(average)

test = loadfromcsv(traindatapath2)

def calmape(data, test):
	acc_r = 0
	for route in routes:
		acc_w, num_w = 0, 0
		for weekday in weekdays:
			for ampm in ampms.keys():
				predict = data.groupby(['route','weekday','am-pm']).get_group((route,weekday,ampm))[['time-interval','avg-time']]
				actual = test.groupby(['route','weekday','am-pm']).get_group((route,weekday,ampm))[['time-interval','avg-time']]

				dic = {}
				for idx, row in predict.iterrows():
					dic[row['time-interval']] = row['avg-time']

				for idx, row in actual.iterrows():
					if row['time-interval'] in ampms[ampm][-6:]:
						acc_w += abs(dic[row['time-interval']]-row['avg-time'])/row['avg-time']
						num_w += 1
		acc_r += acc_w/num_w
	return acc_r/len(routes)

print ("lasso - %f" %(calmape(lassodata, test)))
print ("arima - %f" %(calmape(arimadata, test)))
print ("average - %f" %(calmape(averagedata, test)))
exit(-1)

def combinemodels():
	X, y = [], []

	averagedata.rename(columns={'avg-time':'average'}, inplace=True)
	lassodata.rename(columns={'avg-time':'lasso'}, inplace=True)
	arimadata.rename(columns={'avg-time':'arima'}, inplace=True)

	result = pd.merge(averagedata, lassodata, how='inner', on=['route','weekday','am-pm','time-interval'])
	result = pd.merge(result, arimadata, how='inner', on=['route','weekday','am-pm','time-interval'])

	result['avg-time'] = (result['average']+result['lasso']+result['arima'])/3
	print (calmape(result, test))
	exit(-1)

	actual = test[['route','weekday','am-pm','time-interval','avg-time']]

	combined = pd.merge(result, actual, how='inner', on=['route','weekday','am-pm','time-interval'])

	for idx, row in combined[['route','weekday','am-pm','time-interval','average','lasso','arima','avg-time']].iterrows():
		X.append([row['average'],row['lasso'],row['arima']])
		y.append(row['avg-time'])

	return X, y

from sklearn import linear_model
X, y = combinemodels()

# reg = linear_model.Lasso(alpha = alpha)
# reg.fit(X, y)