from sklearn import linear_model
from commons import *
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

rawdata = loadfromcsv(traindatapath)
testdata = loadfromcsv(testdatapath)

rawdata = rawdata.append(testdata)
import datetime
rawdata = rawdata[((rawdata.datetime < datetime.datetime(2016,10,1)) | 
            (rawdata.datetime > datetime.datetime(2016,10,7))) &
            ((rawdata.datetime < datetime.datetime(2016,9,15)) |
            (rawdata.datetime > datetime.datetime(2016,9,17)))]

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

# generate submit version
for ampm in ampms.keys():
	for route in routes:
		for weekday in weekdays:
			# log-smooth
			sub = np.log(np.array(data[route][weekday][ampm]))
			tar = sub[-1,:6]

			for remain in range(6,12):
				X, y = sub[:-1,:remain], sub[:-1,remain]

				reg = linear_model.Lasso(alpha = 0.1)
				reg.fit(X, y)

				tar = np.append(tar, reg.predict(tar)[0])
			# print ("%s %s %s" %(route, weekday, np.exp(tar[6:])))
			for pred in np.nditer(np.exp(tar[6:])):
				print (pred)