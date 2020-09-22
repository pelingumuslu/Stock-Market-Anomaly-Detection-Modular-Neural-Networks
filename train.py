#Created by Pelin Gümüşlü

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.metrics import sparse_categorical_crossentropy,binary_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.utils import to_categorical
from mnn import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from dataProcess import *
#data with no error must be target data
#normal data
#xtrainde önce 16 tane feature verilir 17.label olur elle anormal ya da  normal olarak belirtilir

#epochs -->iterations over dataset to train for
#batch size --> number of samples per gradient update
#optimizer & loss function && metrics

#if Y(i)s are one-hot encoded then use sparse categorical loss function
#if Y(i)s are integers then use categorical loss function

import matplotlib
import matplotlib.pylab as plt


#instrument,symbol,open,high,low,close,settle price,timestamp
def tr_model1(data):
	return vMerge(vMerge(data[:,0:2],data[:,5:10]),data[:,14:15])
#instrument,symbol,openinterest,changeinIO,valinlakh,timestamp
def tr_model2(data):
	return vMerge(data[:,:5],data[:,11:15])
#instrument,symbol,expirydate,strikeprc,optiontype,contracts
def tr_model3(data):
	return vMerge(data[:,:5],data[:,10:11])

def runModels(train_x,train_y,test_x,test_y):
	"""
	trainx1=tr_model1(train_x)
	trainx2=tr_model2(train_x)
	trainx3=tr_model3(train_x)
	
	testx1=tr_model1(test_x)
	testx2=tr_model2(test_x)
	testx3=tr_model3(test_x)
	"""
	module1=SubModule1(train_x.shape[1])
	module2=SubModule2(train_x.shape[1])
	module3=SubModule3(train_x.shape[1])
	gate=gatingNet(train_x.shape[1])

	historyM1,pred1 = modelfit(module1,train_x,train_y,test_x,test_y)	
	historyM2,pred2 = modelfit(module2,train_x,train_y,test_x,test_y)
	historyM3,pred3 = modelfit(module3,train_x,train_y,test_x,test_y)	
	historyGT,predgt = gatefit(gate,train_x,train_y,test_x,test_y)
	"""

	module1=SubModule1(trainx1.shape[1])
	module2=SubModule2(trainx2.shape[1])
	module3=SubModule3(trainx3.shape[1])
	gate=gatingNet(train_x.shape[1])

	historyM1,pred1 = modelfit(module1,trainx1,train_y,testx1,test_y)	
	historyM2,pred2 = modelfit(module2,trainx2,train_y,testx2,test_y)
	historyM3,pred3 = modelfit(module3,trainx3,train_y,testx3,test_y)	
	historyGT,predgt = gatefit(gate,train_x,train_y,test_x,test_y)
	"""
	connectModules(test_x,pred1,pred2,pred3,predgt)

	return
	
def connectModules(test_x,pred1,pred2,pred3,predgt):
	length = len(pred1)
	
	Sum=np.array([])
	Sum2=np.array([])

	
	for i in range(0,length):
		sum1 = (pred1[i][0]*predgt[i][0]) + (pred2[i][0]*predgt[i][1]) + (pred3[i][0]*predgt[i][2])#label 0 probabilistic
		sum2 = (pred1[i][1]*predgt[i][0]) + (pred2[i][1]*predgt[i][1]) + (pred3[i][1]*predgt[i][2])#label 1 probabilistic
		

		Sum=np.append(Sum,sum1)
		Sum2=np.append(Sum2,sum2)
	#plt.plot(train_x,'r.')
	#plt.plot(Sum,'y.')
	
	
	fig=plt.figure()
	plt.xlabel("Value Index")
	plt.ylabel("Anomaly Scores(%)")
	plt.plot(Sum2*100,'g.')
	plt.show()
	
	#Sum=vMerge(Sum,Sum2)
	
	return
	
	
def gatefit(Model,train_x,train_y,test_x,test_y):
	
	Model.summary()
	print("gate module summary has been published above")
	Model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['binary_crossentropy'])
	
	history = Model.fit(train_x,train_y, epochs=10, batch_size=128,verbose=2,validation_split=0.2)	
	#result = Model.evaluate(test_x,test_y,verbose=2,batch_size=128)
	predict = Model.predict(test_x)

	return history, predict

def modelfit(Model,train_x,train_y,test_x,test_y):
	Model.summary()
	print("module summary has been published above")
		
	Model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['binary_crossentropy'])
	#Model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy')
	#Model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	
	history = Model.fit(train_x,train_y, epochs=10, batch_size=128,verbose=2,validation_split=0.2)
	#result = Model.evaluate(test_x,test_y,verbose=2,batch_size=128)
	predict = Model.predict(test_x)

	return history,predict
	


