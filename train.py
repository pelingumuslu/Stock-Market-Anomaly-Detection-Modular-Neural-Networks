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

def train(train_x,train_y):
	
	#traindata=pd.read_csv("combined2.csv")

	#train1=traindata[['INSTRUMENT','SYMBOL','EXPIRY_DT','OPTION_TYP','TIMESTAMP']]
	train_x1=traindata[['OPEN','CLOSE','SETTLE_PR']]
	train_x2=traindata[['OPEN','HIGH','LOW','CLOSE']]
	train_x3=traindata[['STRIKE_PR','OPEN','HIGH','LOW','CLOSE','SETTLE_PR','OPEN_INT','CHG_IN_OI']]
	train_x4=traindata[['OPEN_INT','CHG_IN_OI']]
	train_x5=traindata[['STRIKE_PR','SETTLE_PR']]
	train_x6=traindata[['CONTRACTS','OPEN_INT','CHG_IN_OI']]
	train_x7=traindata[['SETTLE_PR','VAL_INLAKH']]
	
	#train_y=np.array(traindata[['ANOMALYSCORE']])
	
	train_x1=np.array(train_x1)
	train_x2=np.array(train_x2)
	train_x3=np.array(train_x3)
	train_x4=np.array(train_x4)
	train_x5=np.array(train_x5)
	#these fields are string&datetime values 
	#neural network structures are only accept
	#float or integer values
	inn=traindata[['INSTRUMENT']]
	sym=traindata[['SYMBOL']]
	exp=traindata[['EXPIRY_DT']]
	op=traindata[['OPTION_TYP']]
	tms=traindata[['TIMESTAMP']]
	
	#now we are encoding string values to integers to feed our network
	le=LabelEncoder()
	#ohe=OneHotEncoder()
	#inOhe=ohe.fit_transform(np.ravel(inn))
	inn=le.fit_transform(np.ravel(inn))
	sym=le.fit_transform(np.ravel(sym))
	exp=le.fit_transform(np.ravel(exp))
	op=le.fit_transform(np.ravel(op))
	tms=le.fit_transform(np.ravel(tms))
	
	#inn=to_categorical(inn)
	#sym=to_categorical(sym)
	#print(inn.shape)
	#print(sym.shape)
	mergedInstSym=np.vstack((inn,sym))
	mergedInstSym=mergedInstSym.transpose()
	#print(mergedInstSym)
	mergedExpTmStm=np.vstack((exp,tms))
	mergedExpTmStm=	mergedExpTmStm.transpose()
	
	
	#each module will be fed by different feature combination
	#here is the combinations that I've made	
	train1=np.column_stack((mergedInstSym,train_x1))
	
	train2=np.column_stack((mergedInstSym,train_x4))
	train2=np.column_stack((train2,tms))
	
	train3=np.column_stack((train_x2,mergedExpTmStm))
	
	train4=np.column_stack((op,train_x6))

	train5=np.column_stack((mergedInstSym,train_x7))
	
	
	#transformer = ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[0,1,2,3,4])],remainder='passthrough')
	#transformer3 = ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[2,15])],remainder='passthrough')
	#traindata3=np.array(transformer3.fit_transform(traindata))	
	#OneHotEncoder(categories='auto',dtype='numpy.float64',handle_unknown='error',n_values=None,sparse=True)

			
	#modules
	module1=SubModule1(train1.shape[1])
	module2=SubModule2(train2.shape[1])
	module3=SubModule3(train3.shape[1])
		
	
	modelfit(module1,train1,train_y)	
	modelfit(module2,train2,train_y)
	modelfit(module3,train3,train_y)	
	
	
	return
	
#instrument,symbol,open,high,low,close,settle price
def tr_model1(data):
	return vMerge(data[:,0:2],data[:,5:10])
#instrument,symbol,openinterest,changeinIO,valinlakh,timestamp
def tr_model2(data):
	return data[:,11:15]
#instrument,symbol,expirydate,strikeprc,optiontype,contracts
def tr_model3(data):
	return vMerge(data[:,2:5],data[:,10:11])


def runModels(train_x,train_y,test_x,test_y):
	trainx1=tr_model1(train_x)
	trainx2=tr_model2(train_x)
	trainx3=tr_model3(train_x)
	
	testx1=tr_model1(test_x)
	testx2=tr_model2(test_x)
	testx3=tr_model3(test_x)

	module1=SubModule1(trainx1.shape[1])
	module2=SubModule2(trainx2.shape[1])
	module3=SubModule3(trainx3.shape[1])
	
	gate=gatingNet(train_x.shape[1])
	
	historyM1 = modelfit(module1,trainx1,train_y,testx1,test_y)	
	historyM2 = modelfit(module2,trainx2,train_y,testx2,test_y)
	historyM3 = modelfit(module3,trainx3,train_y,testx3,test_y)	
	
	historyGT = gatefit(gate,train_x,train_y,test_x,test_y)
	
	
	
	#connectAll(historyM1,historyM2,historyM3,historyGT)
	
	return
	
	
def gatefit(Model,train_x,train_y,test_x,test_y):
	
	Model.summary()
	print("gate module summary has been published above")
	Model.compile(optimizer=Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['binary_crossentropy'])
	
	history = Model.fit(train_x,train_y, epochs=20, batch_size=128,verbose=2,validation_split=0.1)
	
	result = Model.evaluate(test_x,test_y,verbose=2,batch_size=128)
	
	
	train_loss = history.history['loss']
	val_loss   = history.history['val_loss']
	train_acc  = history.history['binary_crossentropy']
	#acc    = history.history['val_acc']
	xc         = range(20)
	

	figure = plt.figure()
	plt.xlabel("per epoch")
	plt.ylabel("loss")
	plt.plot(xc, train_loss,'r')
	plt.plot(xc, val_loss,'b')
	plt.legend()
	plt.show()
	
	figure2 = plt.figure()
	plt.xlabel("per epoch")
	plt.ylabel("accuracy")
	#plt.plot(xc, acc,'r')
	plt.plot(xc,train_acc,'g')
	plt.legend()
	plt.show()
	
	
	return
	


def modelfit(Model,train_x,train_y,test_x,test_y):
	Model.summary()
	print("module summary has been published above")
	
	
	Model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['binary_crossentropy'])
	
	#Model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy')
	#Model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	
	history = Model.fit(train_x,train_y, epochs=10, batch_size=128,verbose=2,validation_split=0.2)

	result = Model.evaluate(test_x,test_y,verbose=2,batch_size=128)

	train_loss = history.history['loss']
	val_loss   = history.history['val_loss']
	train_acc  = history.history['binary_crossentropy']
	#val_acc    = history.history['val_acc']
	xc         = range(10)
	
	
	figure = plt.figure()
	plt.xlabel("per epoch")
	plt.ylabel("loss")
	plt.plot(xc, train_loss,'r')
	plt.plot(xc, val_loss,'b')
	plt.legend()
	plt.show()
	
	figure2 = plt.figure()
	plt.xlabel("per epoch")
	plt.ylabel("accuracy")
	#plt.plot(xc, acc,'r')
	plt.plot(xc,train_acc,'g')
	plt.legend()
	plt.show()
	
	#output = K.sum(expert_outputs * K.repeat_elements(K.expand_dims(gating_outputs, axis=1), self.units, axis=1), axis=2)

	return history 
	


