#Created by Pelin Gümüşlü

#In our model each sub-module fed by spesific input combination rather than whole features
#It is determined by me,thus architecture is changeable due to our feature types
#Hidden layers are activated with relu function 
#Output layer is activated with sigmoid function


from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.layers.merge import concatenate

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

def ReLuHiddenLayer(num,prevLayer):
	#hidden layer is created as the given number 
	#of neurons and connected to previous layer 
	hiddenLayer = Dense(num,activation='relu')(prevLayer)
	return hiddenLayer

def SigmHiddenLayer(num,prevLayer):
	hiddenLayer = Dense(num,activation='sigmoid')(prevLayer)
	return hiddenLayer
	
def OutputLayer(prevLayer):
	outLayer = Dense(2,activation='softmax')(prevLayer)
	return outLayer

#Submodules
def SubModule1(inputsize):
	inlayer=InputLayer(inputsize)
	hidden1 = ReLuHiddenLayer(inputsize-1,inlayer)
	hidden2 = ReLuHiddenLayer(inputsize-2,hidden1)
	out = OutputLayer(hidden2)
	module1 = Model(inputs=inlayer,outputs=out)
	print("module 1 successfully created")
	return module1

def SubModule2(inputsize):
	inlayer=InputLayer(inputsize)
	hidden1 = ReLuHiddenLayer(inputsize-1,inlayer)
	hidden2 = ReLuHiddenLayer(inputsize-2,hidden1)
	out = OutputLayer(hidden2)
	module2 = Model(inputs=inlayer,outputs=out)
	print("module 2 successfully created")
	return module2
	
def SubModule3(inputsize):
	inlayer=InputLayer(inputsize)
	hidden1 = ReLuHiddenLayer(inputsize-1,inlayer)
	hidden2 = ReLuHiddenLayer(inputsize-2,hidden1)
	out = OutputLayer(hidden2)
	module3 = Model(inputs=inlayer,outputs=out)
	print("module 3 successfully created")
	return module3
	


#prunning during training




