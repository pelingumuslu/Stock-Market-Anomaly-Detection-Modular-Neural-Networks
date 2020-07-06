from test import *
from mnn import *	
from train import *
from dataProcess import *


if __name__ == '__main__':
	
	data = readCsv()
	
	train_x=data[:12000,1:]
	test_x=data[13000:16000,1:]
	train_y=getAnomalyScores(train_x)
	test_y=getAnomalyScores(test_x)

	runModels(train_x,train_y,test_x,test_y)
	
