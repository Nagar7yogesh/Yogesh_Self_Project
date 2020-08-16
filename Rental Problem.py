import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt


def mean_squared_loss(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    error= ydata-y_predict
    sum_error=0
    for i in range(len(error)):
        sum_error+=pow(error[i],2)
    sum_error=(sum_error)/len(xdata);
    return sum_error

def mean_squared_gradient(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    error= ydata-y_predict
    gradient=-(1.0/len(xdata))*error.dot(xdata)
    return gradient

def mean_absolute_loss(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    error= abs(ydata-y_predict)
    sum_error=sum(error)
    sum_error=sum_error/len(xdata);
    
    return sum_error

def mean_absolute_gradient(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    error= ydata-y_predict
    error_sign=np.sign(error)
    gradient=-(1.0/len(xdata))*error_sign.dot(xdata)
    return gradient

def mean_log_cosh_loss(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    err=abs(y_predict-ydata)
    for i in range(len(err)):
        if abs(err[i])>200 :
            err[i]=200
    error= np.log(np.cosh(err))
    sum_error=sum(error)
    sum_error=sum_error/len(xdata);
    
    return sum_error

def mean_log_cosh_gradient(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    error= ydata-y_predict
    tanhp=np.tanh(error)
    gradient=-(1.0/len(xdata))*tanhp.dot(xdata)
    return gradient

def root_mean_squared_loss(xdata, ydata, weights):
    abc=np.sqrt(mean_squared_loss(xdata, ydata, weights))
    return abc

def root_mean_squared_gradient(xdata, ydata, weights):
    y_predict=np.matmul(xdata,weights)
    error= ydata-y_predict
    errsqrt= 1.0/np.sqrt(error.dot(error.transpose()))
    gradient=-(1.0/np.sqrt(len(xdata)))*error.dot(xdata)*errsqrt
    return gradient

class LinearRegressor:
    weights=[]
    error=[]
    
    def __init__(self,dims):
        self.weights=[]
        for i in range(dims):
            self.weights.append(1)
        self.weights[26]=0
        self.weights[27]=0
        self.weights=np.array(self.weights)
            
		

    def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=500, lr=0.250):
        self.error=[]
        mean_square_error=[]
        for i in range(1,epoch):
            gradient=globals()[gradient_function](xtrain,ytrain,self.weights)
            self.weights=self.weights-(lr)*gradient
            self.error.append(globals()[loss_function](xtrain,ytrain,self.weights))
            mean_square_error.append(mean_squared_loss(xtrain,ytrain,self.weights))
            if i%10==0:
                print('mean square error after epoch',i,'is',mean_squared_loss(xtrain,ytrain,self.weights))
        print('Error for given loss function is'+str(globals()[loss_function](xtrain,ytrain,self.weights)))
        print('mean square error is'+str(mean_squared_loss(xtrain,ytrain,self.weights)))
        self.ploterror( mean_square_error)
        

    def predict(self, xtest):
        alku=[]
        alku.append(['instance (id)','count'])
        y_predicted=np.matmul(xtest,self.weights)
        for i in range(0,len(y_predicted)):
            if(y_predicted[i]<0):
                y_predicted[i]=12
            else:
                y_predicted[i]=int(y_predicted[i])
        for i in range(0,len(y_predicted)):
            alku.append([i,y_predicted[i]])
            
        with open('Prediction.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(alku)
        return y_predicted
   
    def ploterror(self,mean_square_error):
         count=[]
         for i in range(len(self.error)):
             count.append(i)
         plt.plot(count, self.error, color='r')
         plt.plot(count,mean_square_error, color='b')
         plt.xlabel('Epocs :'+str(len(self.error)))
         plt.ylabel('Error')
      
         

def read_dataset(trainfile, testfile):
	'''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
	xtrain = []
	ytrain = []
	xtest = []

	with open(trainfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtrain.append(row[:-1])
			ytrain.append(row[-1])

	with open(testfile,'r') as f:
		reader = csv.reader(f,delimiter=',')
		next(reader, None)
		for row in reader:
			xtest.append(row)

	return xtrain, ytrain, xtest

def preprocess_dataset(xdata, ydata=None):
    xpd=[]
    for i in range(0,len(xdata)):
        xdata[i][1] =(xdata[i][1][0:1])
        if (xdata[i][5] == 'Monday'):
            xdata[i][5] = 1
        if (xdata[i][5] == 'Tuesday'):
            xdata[i][5] = 2
        if (xdata[i][5] == 'Wednesday'):
            xdata[i][5] = 3
        if (xdata[i][5] == 'Thursday'):
            xdata[i][5] = 4
        if (xdata[i][5] == 'Friday'):
            xdata[i][5] = 5
        if (xdata[i][5] == 'Saturday'):
            xdata[i][5] = 6
        if (xdata[i][5] == 'Sunday'):
            xdata[i][5] = 7
            
    for i in range(0, len(xdata)):
       a=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
       a[int(xdata[i][2])-1]=1
       a[4+int(int(xdata[i][3])%24)]=1
       a[27]=int(xdata[i][4])
       a[27+int(xdata[i][5])]=1
       a[34]=int(xdata[i][6])
       a[34+int(xdata[i][7])]=1
       a[38]=xdata[i][8]
       a[39]=xdata[i][9]
       a[40]=xdata[i][10]
       a[41]=xdata[i][11]
       a[42+int(int(xdata[i][1])%10)]=1
       xpd.append(a)
    
    
       
    for i in range(0, len(xpd)):
        for j in range (0, len(xpd[0])):
            xpd[i][j]=float(xpd[i][j])

    if(ydata==None):
      return np.array(xpd)
    else:
      for i in range(0, len(ydata)):
         ydata[i] = float(ydata[i])

      return np.array(xpd),np.array(ydata)
  

dictionary_of_losses = {
	'mse':('mean_squared_loss', 'mean_squared_gradient'),
	'mae':('mean_absolute_loss', 'mean_absolute_gradient'),
	'rmse':('root_mean_squared_loss', 'root_mean_squared_gradient'),
	'logcosh':('mean_log_cosh_loss', 'mean_log_cosh_gradient'),
}

def main():
    xtrain, ytrain, xtest = read_dataset('train.csv', 'test.csv')
    xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)  
    xtestprocessed = preprocess_dataset(xtest)
    dims=len(xtrainprocessed[0])
    print('number of data pts :=' + str(dims))
    loss_fn, loss_grad = dictionary_of_losses['mse']
    model2 = LinearRegressor(dims)
    
    model2.train(xtrainprocessed, ytrainprocessed, loss_fn, loss_grad, 500,0.25)
    
    
    ytest = model2.predict(xtestprocessed)
    print(ytest)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--loss', default='mse', choices=['mse','mae','rmse','logcosh'], help='loss function')
	parser.add_argument('--lr', default=1.0, type=float, help='learning rate')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--train_file', type=str, help='location of the training file')
	parser.add_argument('--test_file', type=str, help='location of the test file')

	args = parser.parse_args()

	main()
