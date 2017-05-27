from commons import *
import numpy as np
import matplotlib.pyplot as plt

# phrase-1
# rawdata = loadfromcsv(traindatapath)

# phrase-2
rawdata = loadfromcsv(phrase2train)

import datetime
rawdata = rawdata[((rawdata.datetime < datetime.datetime(2016,10,1)) | 
            (rawdata.datetime > datetime.datetime(2016,10,7))) &
            ((rawdata.datetime < datetime.datetime(2016,9,15)) |
            (rawdata.datetime > datetime.datetime(2016,9,17)))]

for ampm in ampms.keys():
	for route in routes:
		for weekday in weekdays:
			grouped = rawdata.groupby(['route','weekday','am-pm']).get_group((route,weekday,ampm))[['date','time-interval','avg-time']]

			# for miss values
			alter = grouped.groupby('time-interval')['avg-time'].mean()
			for x in alter.values.tolist()[-6:]:
				print (x)