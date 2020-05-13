from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
from sklearn import linear_model
from datetime import datetime
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import copy

stock_of_interest = 'LEA'
related_stocks = ['THRM', 'GM', 'F', 'STRT', 'HUN', 'GTX', 'ULH', 'FCA']
START_DATE = '2019-01-01'
END_DATE = str(datetime.now().strftime('%Y-%m-%d'))
window = 8
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
	for i in range(len(s[stock_of_interest])-delay):
		s[stock_of_interest][i] = s[stock_of_interest][i+delay]
	s.drop(s.tail(delay).index,inplace=True)
	return s

def select_data(s, i, size, price_target):
	#get the current price of the day
	s.drop(s.head(i).index,inplace=True)
	s.drop(s.tail(size-window-i-1).index,inplace=True)
	r = s.iloc[-1]
	s.drop(s.tail(1).index,inplace=True)
	price_target[0] = s[stock_of_interest][-delay]
	return s, r

def model(s, r, price_target):
	reg = linear_model.LinearRegression()
	reg.fit(s[related_stocks], s[stock_of_interest])
	c = reg.score(s[related_stocks], s[stock_of_interest])
	p = []
	for x in related_stocks:
		p.append(r[x])
	price_target[delay] = reg.predict([p])
	return c

def trade(c, profit, shares, price_target):
	dif = price_target[1] - price_target[0]
	if c > .92 and dif > 2:
		if shares == 0:
			#buy shares
			shares += 1
			profit -= price_target[0]
	else:
		if shares != 0:
			#sell shares
			shares -= 1
			profit += price_target[0]
	return profit, shares
			

def main(): 
	#fetch data
	df = load_data()
	size = len(df)
	profit = 0
	shares = 0
	price_target = [0]
	for x in range(delay):
		price_target.append(0); 

	for i in range(size-window-1):
		#preprocess data
		s = copy.deepcopy(df)
		s, r = select_data(s, i, size, price_target)

		#run through prediction 
		c = model(s, r, price_target)

		#decide wether to buy or sell
		profit, shares = trade(c, profit, shares, price_target)

		#shift price target
		for i in range(delay-1):
			price_target[i+1] = price_target[i+2]

	#sell any remaining shares
	if shares != 0:
		profit += shares * price_target[0]

	print(profit)

if __name__ == "__main__": 
	main()