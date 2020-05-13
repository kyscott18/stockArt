from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError
from sklearn import linear_model
from datetime import datetime
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import copy
import math
import csv

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
	se = 0; 
	for i in range(window):
		p = []
		for x in related_stocks:
			p.append(s[x][i])
		se += (reg.predict([p]) - s[stock_of_interest][i])**2
	se = se/(window-2)
	se = math.sqrt(se)
	p = []
	for x in related_stocks:
		p.append(r[x])
	price_target[delay] = reg.predict([p])
	return se

def trade(se, profit, shares, price_target, writer):
	#only sell if confident price will fall
	low  = price_target[1] - (2.807 * se)
	high = price_target[1] + (2.807 * se)
	if low > price_target[0]:
		#buy shares
		shares += 1
		profit -= price_target[0]	
		writer.writerow({'type': 'buy', 'shares': '1', 'price': price_target[0], 'low prediction': low, 'prediction': price_target[1], 'high prediction': high})	
	elif high < price_target[0]:
		if shares != 0:
			#sell shares
			profit += price_target[0] * shares
			writer.writerow({'type': 'sell', 'shares': shares, 'price': price_target[0], 'low prediction': low, 'prediction': price_target[1], 'high prediction': high})	
			shares = 0
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
	with open('trade_data.csv', mode='w') as csv_file:
		fieldnames = ['type', 'shares', 'price', 'low prediction', 'prediction', 'high prediction']
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()

		for i in range(size-window-1):
			#preprocess data
			s = copy.deepcopy(df)
			s, r = select_data(s, i, size, price_target)

			#run through prediction 
			se = model(s, r, price_target)

			#decide wether to buy or sell
			profit, shares = trade(se, profit, shares, price_target, writer)

			#shift price target
			for i in range(delay-1):
				price_target[i+1] = price_target[i+2]

	#sell any remaining shares
	if shares != 0:
		print(shares)
		profit += shares * price_target[0]

	print(profit)

if __name__ == "__main__": 
	main()