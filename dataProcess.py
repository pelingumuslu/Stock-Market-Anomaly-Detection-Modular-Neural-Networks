#Created by Pelin Gümüşlü
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt

from mnn import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.utils import to_categorical

seqNum = np.array([])
instrument = np.array([])
symbol = np.array([])
expDate = np.array([])
strikePrice = np.array([])
optionType = np.array([])
openVal = np.array([])
highVal = np.array([])
lowVal = np.array([])
closeVal = np.array([])
settlePrice = np.array([])
contracts = np.array([])
valInlakh = np.array([])
openInt = np.array([])
changeInIO = np.array([])
timeStamp = np.array([])

#whole data
def readCsv():
	filename = "NSE.csv"
	data=pd.read_csv(filename)
	
	global seqNum,instrument,symbol,expDate,strikePrice,optionType,openVal,highVal,lowVal,closeVal,settlePrice,contracts,valInlakh,openInt,changeInIO,timeStamp
	
	seqNum = np.append(seqNum, data['SEQNUM'])
	"""
	instrument = encoder(np.append(instrument, data['INSTRUMENT']))
	symbol = encoder( np.append(symbol, data['SYMBOL']))
	expDate = encoder(np.append(expDate, data['EXPIRY_DT']))
	"""
	instrument = np.append(instrument, data['INSTRUMENT'])
	symbol = np.append(symbol, data['SYMBOL'])
	expDate = np.append(expDate, data['EXPIRY_DT'])
		
	strikePrice = np.append(strikePrice, data['STRIKE_PR'])

	optionType = np.append(optionType, data['OPTION_TYP'])
	
	openVal = np.append(openVal,data['OPEN'])
	highVal = np.append(highVal,data['HIGH'])
	lowVal = np.append(lowVal, data['LOW'])
	closeVal = np.append(closeVal, data['CLOSE'])
	settlePrice = np.append(settlePrice, data['SETTLE_PR'])
	contracts = np.append(contracts, data['CONTRACTS'])
	valInlakh = np.append(valInlakh, data['VAL_INLAKH'])
	openInt = np.append(openInt, data['OPEN_INT'])
	changeInIO = np.append(changeInIO, data['CHG_IN_OI'])	
	
	timeStamp = np.append(timeStamp, data['TIMESTAMP'])
	
	
	instrument = onehotencoder(instrument)
	symbol = onehotencoder(symbol)
	expDate = onehotencoder(expDate)
	optionType = onehotencoder(optionType)
	timeStamp = onehotencoder(timeStamp)
	"""
	instrument2 = to_categorical(np.array(data['INSTRUMENT']))
	symbol2 = to_categorical(data['SYMBOL'])
	expDate2 = to_categorical(data['EXPIRY_DT'])
	optionType2 = to_categorical(data['OPTION_TYP'])
	timeStamp2 = to_categorical(data['TIMESTAMP'])
	
	data=np.array(data)
	encoded = to_categorical(data)
		"""
	dataWhole = np.column_stack((seqNum,instrument,symbol,expDate,strikePrice,optionType,openVal,highVal,lowVal,closeVal,settlePrice,contracts,valInlakh,openInt,changeInIO,timeStamp))

	#dataWhole = np.column_stack((seqNum,strikePrice,openVal,highVal,lowVal,closeVal,settlePrice,contracts,valInlakh,openInt,changeInIO))
	
	return dataWhole

#string values are converted to integers
def encoder(dataframe):
	labelEncoder=LabelEncoder()
	return labelEncoder.fit_transform(np.ravel(dataframe))

def onehotencoder(dataframe):
	onehotencoder=OneHotEncoder()
	encoded=onehotencoder.fit_transform(np.reshape(dataframe,(-1,1))).toarray()
	print("encoded data")
	np.set_printoptions(threshold=np.inf)
	return encoded

#gets transpose of a matrix
def getTranspose(dataframe):
	return dataframe.transpose()

#merge arrays vertically in given order
#add right one to next column left
def vMerge(df1,df2):
	return np.column_stack((df1,df2))

def anomalyScore(op,cl,stlPrc,opnint,chg,chgIOmean,openIntmean,setPrcmean,clsmean,opnmean,chgIOStd,openIntStd,opnStd,clsStd,setPrcStd):
	score=0
	score+=abs(op-opnmean)/opnStd
	score+=abs(cl-clsmean)/clsStd
	score+=abs(stlPrc-setPrcmean)/setPrcStd
	score+=abs(opnint-openIntmean)/openIntStd
	score+=abs(chg-chgIOmean)/chgIOStd
	print(score/5)
	return score/5

def stdDev(array):
	return np.std(array)
	
def meanValue(array):
	return np.mean(array)
	
def Max(array):
	return np.amax(array)
	
def Min(array):
	return np.amin(array)

def getAnomalyScores(data):
	scores=np.array([])
	
	#z-Scores of each feature
	
	for i in range(0,len(data)):
		zScore=0
		#inst sym exp d
		zScore+=(data[i:i+1,4:5]-meanValue(strikePrice))/stdDev(strikePrice)
		#opttype
		zScore+=(data[i:i+1,6:7]-meanValue(openVal))/stdDev(openVal)
		zScore+=(data[i:i+1,7:8]-meanValue(highVal))/stdDev(highVal)
		zScore+=(data[i:i+1,8:9]-meanValue(lowVal))/stdDev(lowVal)
		zScore+=(data[i:i+1,9:10]-meanValue(closeVal))/stdDev(closeVal)
		zScore+=(data[i:i+1,10:11]-meanValue(settlePrice))/stdDev(settlePrice)
		zScore+=(data[i:i+1,11:12]-meanValue(contracts))/stdDev(contracts)
		zScore+=(data[i:i+1,12:13]-meanValue(valInlakh))/stdDev(valInlakh)
		zScore+=(data[i:i+1,13:14]-meanValue(openInt))/stdDev(openInt)
		zScore+=(data[i:i+1,14:15]-meanValue(changeInIO))/stdDev(changeInIO)
		#timestamp
		scores=np.append(scores,zScore/10)
	return scores
	
"""	
	#mean values
	chgIOmean=meanValue(changeInIO)
	openIntmean=meanValue(openInt)
	setPrcmean=meanValue(settlePrice)
	clsmean=meanValue(closeVal)
	
	#standart variances
	chgIOStd=stdDev(changeInIO)
	openIntStd=stdDev(openInt)
	opnStd=stdDev(openVal)
	clsStd=stdDev(closeVal)
	setPrcStd=stdDev(settlePrice)
	
	
	
	#op,cl,stlPrc,opnint,chg,chgIOmean,openIntmean,setPrcmean,clsmean,opnmean,chgIOStd,openIntStd,opnStd,clsStd,setPrcStd
	
	for i in range(0,len(data)):
		scores=np.append(scores,anomalyScore(data[i:i+1,5:6],data[i:i+1,8:9],data[i:i+1,9:10],data[i:i+1,12:13],data[i:i+1,13:14],chgIOmean,openIntmean,setPrcmean,clsmean,opnmean,chgIOStd,openIntStd,opnStd,clsStd,setPrcStd))
	return scores/100
	#return anomalyScores
"""



