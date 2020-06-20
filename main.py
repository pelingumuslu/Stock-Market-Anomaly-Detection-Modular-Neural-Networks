from test import *
from mnn import *	
from train import *
from dataProcess import *


if __name__ == '__main__':
	#in this section main data is read	
	data = readCsv()
	
	train_x=data[:4000,1:]
	test_x=data[9000:10000,1:]
	
	
	train_y=getAnomalyScores(train_x)
	#train_y=getTranspose(train_y)
	test_y=getAnomalyScores(test_x)
	#test_y=getTranspose(test_y)
	
	
	runModels(train_x,train_y,test_x,test_y)
	
