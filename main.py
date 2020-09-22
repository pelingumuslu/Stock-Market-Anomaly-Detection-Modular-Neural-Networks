
from mnn import *	
from train import *
from dataProcess import *
import timeit


if __name__ == '__main__':
	
	start = timeit.default_timer()

	data = readCsv()
	
	train_x=data[130000:160000,1:]
	test_x=data[170000:176000,1:]
	#train_x=data[30000:40000,1:]
	#test_x=data[70000:72000,1:]
	#train_x=data[:3000,1:]
	#test_x=data[7000:7600,1:]
	
	norm_trainx=normalize(train_x)
	norm_testx=normalize(test_x)

	train_y=getAnomalyScores(train_x)
	test_y=getAnomalyScores(test_x)
	

	runModels(norm_trainx,train_y,norm_testx,test_y)
	
	stop = timeit.default_timer()
	print('Algorithm has trained on ',len(train_x),' and tested on ', len(test_x),' samples')
	print('Time : ', stop - start)

	
	
