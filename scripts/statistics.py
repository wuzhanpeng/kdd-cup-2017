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
		fields[0],
		fields[1],
		str(fields[2].hour)+'-'+str(fields[2].minute/20),
		fields[4],
		'am' if fields[2].hour < 12 else 'pm'), records)

	df = pd.DataFrame(records, columns = ['iid','tid','interval','time','-'])

	return df

def statistic():
	df = loadfromcsv()

	# route classification
	grouped = df.groupby(['iid','tid','interval','-']).mean()

	# print(df)
	# print(df.dtypes)


	# print(ammeancol)
	# print(grouped)

	# pieces = [ammeancol, pmmeancol]
	# res = pd.concat(pieces, axis=1)
	# print(res)

	# a22 = grouped.loc['A','2',:,'am']
	# print(a22)

	# plot
	# a22.plot()

	routes = [('A','2'),('A','3'),('B','1'),('B','3'),('C','1'),('C','3')]
	# routes = [('A','2')]
	# for intersection, tollgate in routes:
	for route in routes:
		df = grouped.loc[route[0],route[1]]
		df.plot(by='-')
		print(df)

	plt.show()

if __name__ == '__main__':
	statistic()