#Created by Pelin Gümüşlü

#In our model each sub-module fed by spesific input combination rather than whole features
#It is determined by me,thus architecture is changeable due to our feature types
#Hidden layers are activated with relu function 
#Output layer is activated with sigmoid function


from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np
from keras.layers.merge import concatenate
from math import floor
#He Initializer is used to assign weights for layers with ReLu activation function
def HeInitializer(layer_size, prevLayer_size):
	weightMatrix = np.random.randn(layer_size,prevLayer_size) *np.sqrt(2/prevLayer_size)
	return weightMatrix

#Xavier initializer is used to assign weights for layers with Tanh activation function
def XavierInitializer():
	weightMatrix = np.random.randn(layer_size,prevLayer_size) *np.sqrt(1/prevLayer_size)
	return weightMatrix

def InputLayer(labels):
	inputLayer = Input(shape=(labels,))
	return inputLayer

def SoftmaxHiddenLayer(num,prevLayer):
	hiddenLayer = Dense(num,activation='softmax')(prevLayer)
	return hiddenLayer

def ReLuHiddenLayer(num,prevLayer):
	#hidden layer is created as the given number 
	#of neurons and connected to previous layer 
	hiddenLayer = Dense(num,activation='relu')(prevLayer)
	return hiddenLayer
	
def OutputLayer(prevLayer):
	outLayer = Dense(2,activation='softmax')(prevLayer)
	return outLayer

#Submodules
def SubModule1(inputsize):
	inlayer = InputLayer(inputsize)
	dropout1=Dropout(0.5)(inlayer)
	hidden1 = ReLuHiddenLayer(inputsize/2,dropout1)
	#dropout1=Dropout(0.4)(hidden1)
	hidden2 = ReLuHiddenLayer(floor(inputsize/4),hidden1)
	out = OutputLayer(hidden2)
	module1 = Model(inputs=inlayer,outputs=out)
	print("module 1 successfully created")
	return module1

def SubModule2(inputsize):
	inlayer = InputLayer(inputsize)
	dropout1=Dropout(0.5)(inlayer)
	hidden1 = ReLuHiddenLayer(inputsize/2,dropout1)
	#dropout1=Dropout(0.4)(hidden1)
	hidden2 = ReLuHiddenLayer(floor(inputsize/4),hidden1)
	out = OutputLayer(hidden2)
	module2 = Model(inputs=inlayer,outputs=out)
	print("module 2 successfully created")
	return module2
	
def SubModule3(inputsize):
	inlayer = InputLayer(inputsize)
	dropout1=Dropout(0.5)(inlayer)
	hidden1 = ReLuHiddenLayer(inputsize/2,dropout1)
	#dropout1=Dropout(0.4)(hidden1)
	hidden2 = ReLuHiddenLayer(floor(inputsize/4),hidden1)
	out = OutputLayer(hidden2)
	module3 = Model(inputs=inlayer,outputs=out)
	print("module 3 successfully created")
	return module3
	
def gatingNet(inputsize):
	inlayer = InputLayer(inputsize)
	hidden1 = ReLuHiddenLayer(inputsize,inlayer)
	#dropout1=Dropout(0.5)(hidden1)
	hidden2 = ReLuHiddenLayer(inputsize,hidden1)
	#dropout2=Dropout(0.4)(hidden2)
	hidden3= ReLuHiddenLayer(floor(inputsize),hidden2)
	#dropout3=Dropout(0.5)(hidden3)
	hidden4= ReLuHiddenLayer(floor(inputsize),hidden3)
	out = 	outLayer = Dense(3,activation='softmax')(hidden4)#number of modules as the output number
	gating = Model(inputs=inlayer,outputs=out)
	return gating
#prunning during training



