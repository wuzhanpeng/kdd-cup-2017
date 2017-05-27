# -*- coding: utf-8 -*-

from datetime import datetime
from commons import *
from arima_pre import *
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA

import warnings
warnings.filterwarnings("ignore", category=Warning)

data = datapreprocess()

output_lst = []
# generate submit version
for ampm in ampms.keys():
	for route in routes:
		for weekday in weekdays:
			# log-smooth
			sub = np.log(tolist(data[route][weekday][ampm]))
			# sub = np.log(tolist(data['C-3'][6]['am']))
			# fit best model
			order = st.arma_order_select_ic(sub,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])

			model = ARMA(sub, order=order.bic_min_order)
			result_arma = model.fit(disp=-1, method='css')
			predict = result_arma.predict()

			start = len(sub) - len(predict)
			end = start + len(predict) + 6
			# fig = result_arma.plot_predict(start, end)
			# fig.suptitle("%s %s %s" %(route,weekday,ampm))

			forecast = result_arma.predict(start, end)[-6:]
			# print (np.exp(sub))
			for x in np.exp(forecast):
				output_lst.append(x)
			# actual = test.get_group((route,weekday,ampm))[-6:]['avg-time'].values.tolist()
			print ("%s %s %s" %(route,weekday,ampm))

			# plt.show()
			# exit(-1)

with open('./arima-forecast2.seq','w') as f:
	for element in output_lst:
		f.write(str(element)+"\n")