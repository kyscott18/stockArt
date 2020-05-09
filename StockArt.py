from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
from collections import deque
from sklearn import linear_model
import matplotlib.pyplot as plt 
from datetime import datetime
import copy
import pandas as pd
import numpy as np

stock_of_interest = 'LEA'
related_stocks = ['THRM', 'GM', 'F', 'STRT', 'HUN', 'GTX', 'ULH', 'FCA']
START_DATE = '2020-01-01'
END_DATE = str(datetime.now().strftime('%Y-%m-%d'))
window = 20
delay = 1

def get_data(ticker):
	try: 
		stock_data = data.DataReader(ticker, 'yahoo', START_DATE, END_DATE)
		return stock_data['Close']
	except RemoteDataError: 
		print('No data found for', ticker)

def load_data():
	s = get_data(stock_of_interest)
	col = [stock_of_interest]
	for x in related_stocks:
		s = pd.concat([s, get_data(x)], axis=1).reindex(s.index)
		col.append(x)
	s.columns = col
	return s

def select_data(s):
	for i in range(len(s[stock_of_interest])-delay):
		s[stock_of_interest][i] = s[stock_of_interest][i+delay]
	s.drop(s.tail(delay).index,inplace=True)
	r = s.iloc[-1]
	s.drop(s.tail(1).index,inplace=True)
	return s, r

def model(s, r):
	reg = linear_model.LinearRegression()
	reg.fit(s[related_stocks], s[stock_of_interest])
	c = reg.score(s[related_stocks], s[stock_of_interest])
	p = []
	for x in related_stocks:
		p.append(r[x])
	dif = reg.predict([p]) - s[stock_of_interest][-1]
	return c, dif

def trade(c, dif):
	print(c, dif)
	if c > .92 and dif > 8:
		print("buy")
	else:
		print("sell")

def main(): 
	#fetch data
	df = load_data()

	#preprocess data
	df, r = select_data(df)

	#run through prediction 
	c, dif = model(df, r)

	#decide wether to buy or sell
	trade(c, dif)

if __name__ == "__main__": 
	main()