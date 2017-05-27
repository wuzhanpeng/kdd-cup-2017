# -*- coding: utf-8 -*-

from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from commons import *
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=Warning)

# phrase-1
# rawdata = loadfromcsv(traindatapath)
# testdata = loadfromcsv(testdatapath)

# phrase-2
rawdata = loadfromcsv(phrase2train)
testdata = loadfromcsv(phrase2test)

rawdata = rawdata.append(testdata)

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
				checked = {}
				# append the miss value
				for interval in ampms[ampm]:
					if interval in raw:
						checked[interval] = raw[interval]
					else:
						if interval in alter.index:
							checked[interval] = alter.loc[interval]
						else:
							checked[interval] = 1e-15

				# refine the value
				for interval in ampms[ampm]:
					if interval in alter.index:
						avg = alter.loc[interval]
						if abs(checked[interval]-avg) > 0.5*avg:
							checked[interval] = avg
						else:
							checked[interval] = (checked[interval]+avg)/2

				tmplst = map(lambda r: checked[r], ampms[ampm])
				data[route][weekday][ampm].append(tmplst)

def datapreprocess():
	return data

def tolist(twodarray):
	lst = []
	for row in twodarray:
		for ele in row:
			lst.append(ele)
	return lst