# Anastasiya Zhukova
# INFO 3401
# Worked with Harold Chang

# Using the candy-data.csv file in the repo, populate an AnalysisData object that will hold 
# the data you'll use for today's problem set. You should read in the data from the CSV, 
# store the data in the dataset variable, and initialize the xs (in your variables attribute) 
# and targetY variables appropriately. targetY should reference the variable describing whether or not a candy is chocolate.


import csv
import pprint
import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score


#didn't know what I was doing here but ended up creating a class that can tell you whether or not a candy is chocolate... whoops
class failedAnalysisData:

	def __init__(self, filename):
		self.filename = filename
		self.dataset = []
		self.feature_names = []
		self.instances = []
		
		with open(filename, 'r') as inFile:
			reader = csv.DictReader(inFile, delimiter = ',', lineterminator = '\n')
			for line in reader:
				self.dataset.append(line)


	def isChocolate(self, candyName):
		data = self.dataset
		#pprint.pprint(data)
		for line in data:
			if line['competitorname'] == candyName and line['chocolate'] == '1':
				print('Yes, ', line['competitorname'], ' is chocolate')
			elif line['competitorname'] == candyName and line['chocolate'] == '0':
				print('No, ', line['competitorname'], ' is not chocolate')

#ad = failedAnalysisData('candy-data.csv')
# ad.isChocolate('Whoppers')
# ad.isChocolate('One dime')



class AnalysisData:
	
	def __init__(self):
		self.variables = []
		self.dataset = []

	def parser(self, filename):
		self.dataset = pd.read_csv(filename, encoding = 'latin1')
		self.variables = self.dataset.columns.values
		#print(self.variables)


try2 = AnalysisData()
try2.parser('candy-data.csv')

dataset = AnalysisData()






# (b) LinearAnalysis, which will contain your functions for doing linear regression and have at 
# a minimum attributes called bestX (which holds the best X predictor for your data),
# targetY (which holds the index to the target dependent variable), 
# and fit (which will hold how well bestX predicts your target variable).

#Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter. 
#Create the same function for LogisticAnalysis. 
#Note that you will use the LinearAnalysis object to try to predict the amount of sugar in the candy 
#and the LogisticAnalysis object to predict whether or not the candy is chocolate.


#https://stackoverflow.com/questions/46092914/sklearn-linearregression-could-not-convert-string-to-float
#https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
#http://www.datasciencemadesimple.com/get-list-column-headers-column-name-python-pandas/
#



class LinearAnalysis:

	def __init__(self, filename):
		self.filename = filename

	def doLinearAnalysis(self):
		candy_df = pd.read_csv(self.filename, encoding='latin1')
		#candy_df.head()

		target = pd.DataFrame(candy_df, columns = ['sugarpercent'])
		X = candy_df
		Y = target['sugarpercent']
		X = X.apply(pd.to_numeric, errors='coerce')
		Y = Y.apply(pd.to_numeric, errors='coerce')
		X.fillna(0, inplace=True)
		Y.fillna(0, inplace=True)

		lm = linear_model.LinearRegression()
		model = lm.fit(X,Y)
		predictions = lm.predict(X)

		print(predictions)




LinAl = LinearAnalysis('candy-data.csv')
LinAl.doLinearAnalysis()





class LogisticAnalysis:

	def __init__(self, filename):
		self.filename = filename

	def doLogisticAnalysis(self):
		candy_df = pd.read_csv(self.filename, encoding='latin1')
		#candy_df.head()

		target = pd.DataFrame(candy_df, columns =['chocolate'])
		X = candy_df
		Y = target['chocolate']
		X = X.apply(pd.to_numeric, errors='coerce')
		Y = Y.apply(pd.to_numeric, errors='coerce')
		X.fillna(0, inplace=True)
		Y.fillna(0, inplace=True)

		lm = linear_model.LogisticRegression()
		model = lm.fit(X,Y)
		predictions = lm.predict(X)
		print(predictions)


LogAl = LogisticAnalysis('candy-data.csv')
LogAl.doLogisticAnalysis()

		