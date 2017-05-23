# -*- coding: utf-8 -*-

from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
from commons import *
import numpy as np
import matplotlib.pyplot as plt

def preprocess(ts):
	ts_log = np.log(ts)
	rol_mean = ts_log.rolling(window=12).mean()
	rol_mean.dropna(inplace=True)
	ts_diff_1 = rol_mean.diff(1)
	ts_diff_1.dropna(inplace=True)
	ts_diff_2 = ts_diff_1.diff(1)
	ts_diff_2.dropna(inplace=True)

	return ts_diff_2, (ts_log,rol_mean,ts_diff_1)

def recover(predict_ts, appendix):
	ts_log,rol_mean,ts_diff_1 = appendix
	diff_shift_ts = ts_diff_1.shift(1)
	diff_recover_1 = predict_ts.add(diff_shift_ts)
	rol_shift_ts = rol_mean.shift(1)
	diff_recover = diff_recover_1.add(rol_shift_ts)
	rol_sum = ts_log.rolling(window=11).sum()
	rol_recover = diff_recover*12 - rol_sum.shift(1)
	log_recover = np.exp(rol_recover)
	log_recover.dropna(inplace=True)

	return log_recover

def display(ts, log_recover):
	ts = ts[log_recover.index]
	plt.figure(facecolor='white')
	log_recover.plot(color='blue', label='Predict')
	ts.plot(color='red', label='Original')
	plt.legend(loc='best')
	print(log_recover-ts)
	plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/ts.size))
	plt.show()

import warnings
warnings.filterwarnings("ignore", category=Warning)

rawdata = loadfromcsv(traindatapath)

b3 = rawdata.groupby(['route','am-pm','weekday']).get_group(('B-3','am',0))[['datetime','avg-time']]
b3.set_index('datetime', inplace = True)

# b3train = b3[:'2016-10-10']
# b3test = b3['2016-10-11':]

import datetime
b3train = pd.DataFrame(b3.values, index=pd.date_range(start=datetime.datetime(2016,1,1), periods=len(b3.index)))
b3train = b3train[0]
# print(b3train)
# print(b3test)

# b3train_log = np.log(b3train)
# print(best_diff(b3train_log, 20))
# exit(-1)

# b3train_log_diff = b3train_log - b3train_log.shift(1)
# print(test_stationarity(b3train_log_diff.dropna()['avg-time']))
# exit(-1)

# preprocess
b3train_log_diff, appendix = preprocess(b3train)

# auto fit best model
import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(b3train_log_diff,max_ar=5,max_ma=5,ic=['aic', 'bic', 'hqic'])
# print ('here %s' % order)

# model = ARIMA(b3train_log, order = (p, 1, q))
# results_ARIMA = model.fit(disp = -1)

# back to original scale
# predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
# predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
# predictions_ARIMA_log = pd.Series(b3train_log.ix[0], index = b3train_log.index)
# predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value = 0)

# predictions_ARIMA = np.exp(predictions_ARIMA_log)
# plt.plot(b3train)
# plt.plot(predictions_ARIMA)

from statsmodels.tsa.arima_model import ARMA
model = ARMA(b3train_log_diff, order=order.bic_min_order)
result_arma = model.fit(disp=-1, method='css')
train_predict = recover(result_arma.predict(), appendix)

# plt.plot(b3train)
# print (b3train)
# print (train_predict)
# plt.plot(train_predict)

# plt.show()

display(b3train, train_predict)