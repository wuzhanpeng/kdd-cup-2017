# -*- coding: utf-8 -*-

from sklearn import linear_model
from commons import *
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

rawdata = loadfromcsv(traindatapath)

import datetime
# rawdata = rawdata[((rawdata.datetime < datetime.datetime(2016,10,1)) | 
#             (rawdata.datetime > datetime.datetime(2016,10,7))) &
#             ((rawdata.datetime < datetime.datetime(2016,9,15)) |
#             (rawdata.datetime > datetime.datetime(2016,9,17)))]

data = {}
for route in routes:
	data[route] = {}
	for weekday in weekdays:
		data[route][weekday] = {}
		for ampm in ampms.keys():
			data[route][weekday][ampm] = []

for route in routes:
	for weekday in weekdays:
		for ampm in ampms.keys():
			grouped = rawdata.groupby(['route','weekday','am-pm']).get_group((route,weekday,ampm))[['date','time-interval','avg-time']]

			# for miss values
			alter = grouped.groupby('time-interval')['avg-time'].mean()

			for date, group in grouped.groupby('date'):
				raw = {}
				for idx, row in group.iterrows():
					raw[row['time-interval']] = row['avg-time']
				checked = []
				for interval in ampms[ampm]:
					if interval in raw:
						checked.append(raw[interval])
					else:
						# print ("loss %s %s %s"% (route,date,interval))
						# print (alter)
						if interval in alter.index:
							checked.append(alter.loc[interval])
						else:
							checked.append(2)
				data[route][weekday][ampm].append(checked)

# local-test
optalpha, minmape = 1, 1
for ratio in range(1,100):
	alpha = 1.0 * ratio / 100
	acc_val = 0
	for route in routes:
		acc_val_w = 0
		for weekday in weekdays:
			for ampm in ampms.keys():
				# log-smooth
				sub = np.log(np.array(data[route][weekday][ampm]))
				tar = sub[-1,:6]

				for remain in range(6,12):
					X, y = sub[:-1,:remain], sub[:-1,remain]

					reg = linear_model.Lasso(alpha = alpha)
					reg.fit(X, y)

					act, pre = np.exp([sub[-1,remain], reg.predict(tar)[0]])
					acc_val_w += abs((act-pre)/act)

					tar = np.append(tar, np.log(pre))
		acc_val += acc_val_w / (7*12)

	mape = acc_val / len(routes)
	print ("iter: %d, alpha: %f MAPE: %f" % (ratio, alpha, mape))
	if mape < minmape:
		optalpha, minmape = alpha, mape

print ("Optimization - alpha: %f MAPE: %f" % (optalpha, minmape))