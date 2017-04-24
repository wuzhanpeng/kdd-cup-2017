# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.style.use('ggplot')

def loadfromcsv():
	agg_csv_file = open('../data/training_20min_avg_travel_time.csv')
	columns = agg_csv_file.readline()

	records = []
	for line in agg_csv_file:
		fields = line.strip("\n").replace('"', '').split(',')
		fields[2] = datetime.strptime(fields[2][1:], "%Y-%m-%d %H:%M:%S")
		fields[3] = datetime.strptime(fields[3][:-1], "%Y-%m-%d %H:%M:%S")
		fields[4] = float(fields[4])
		records.append(fields)

	# filter redundancy time interval
	records = filter(lambda fields: (
		fields[2].day == fields[3].day and (
		fields[2].hour >= 6 and 
		fields[2].hour < 10 or 
		fields[2].hour >= 15 and 
		fields[2].hour < 19)), records)

	records = map(lambda fields: (
		str(fields[0]+'-'+fields[1]),
		str(fields[2].hour)+'-'+str(fields[2].minute/20),
		fields[4],
		'am' if fields[2].hour < 12 else 'pm'), records)

	return records

def statistic():
	records = loadfromcsv()

	# route groups
	routes = {'A-2':[],'A-3':[],'B-1':[],'B-3':[],'C-1':[],'C-3':[]}
	for record in records:
		routes[record[0]].append(record[1:])

	for interval in ['am','pm']:
		i = 1
		for route in routes:
			a2 = routes[route]
			data = filter(lambda r: r[2] == interval, a2)			
			# print(data)

			# basic statistic
			timeSeq = map(lambda r: r[1], data)
			_max = max(timeSeq)
			_min = min(timeSeq)
			_mean = sum(timeSeq)/len(timeSeq)

			# 10-min
			plt.subplot(3,2,i)
			plt.hist(np.array(timeSeq), bins = range(1, 500, 10), alpha = 0.5, label = route)
			plt.legend(loc='upper right')
			plt.title(str(_max)+' | '+str(_mean)+' | '+str(_min))
			i = i + 1

		print("showing: "+interval)
		plt.suptitle(interval, size = 40)
		plt.show()

if __name__ == '__main__':
	statistic()