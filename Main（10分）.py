import math
import datetime
import sys
import numpy as np
import multiprocessing as mp
fr = open( "train_data.txt")
def job(line):
    allInfo = line.strip().split(',')
    temp = np.array(allInfo).astype(float)
    return temp
def DeaLTrain(a):
    line = fr.readline()
    allInfo = line.strip().split(',')
    temp = np.array(allInfo).astype(float)
    return temp
import time 

class LR:
    def __init__(self, train_file_name, test_file_name, predict_result_file_name):
        self.train_file = train_file_name
        self.predict_file = test_file_name
        self.predict_result_file = predict_result_file_name
        self.max_iters = 300
        self.rate = 0.1
        self.feats = []
        self.labels = []
        self.feats_test = []
        self.labels_predict = []
        self.param_num = 0
        self.weight = []

    def loadDataSet(self, file_name):
        feats = []
        labels = []
        fr = open(file_name)
        lines = fr.readlines()
        pool = mp.Pool()
        res = pool.map(job, lines)
        feats = res
        pool.close()
        fr.close()
        print(feats[6666][0],feats[6666][1])
        return feats, labels
        
    def loadDataSet1(self, file_name):
        fp = open(file_name)
        pool = mp.Pool() # 无参数时，使用所有cpu核
        max_read = 1536
        lines = []
        line = fp.readline()
        while line:
            if max_read <=0:
                break
            else:
                max_read -= 1
            lines.append(line)
            line = fp.readline()
        res = pool.map(job, lines)
        '''res = pool.map(DeaLTrain, range(max_read))'''
        res = np.array(res)
        feats = res[:,0:-1]
        labels = res[:,-1]
        pool.close()
        fp.close()
        return feats, labels
    def loadTrainData(self):
        self.feats, self.labels = self.loadDataSet1(self.train_file)
        self.labels = self.labels.reshape(1,len(self.labels))
        #标准化
        self.var = self.feats.var(axis = 0)
        self.mean = self.feats.mean(axis = 0)
        self.feats = (self.feats - self.mean)/self.var
        self.feats = self.feats.T
    def loadTestData(self):   
        self.feats_test, self.labels_predict = self.loadDataSet(self.predict_file)

    def savePredictResult(self):        
        f = open(self.predict_result_file, 'w')
        for i in range(len(self.labels_predict)):
            f.write(str(int(self.labels_predict[i]))+"\n")
        f.close()

    def printInfo(self):
        print(self.train_file)
        print(self.predict_file)
        print(self.predict_result_file)
        print(self.feats)
        print(self.labels)
        print(self.feats_test)
        print(self.labels_predict)
        
    def initParams(self,param_num):
        np.random.seed(1)
        W1 = np.random.rand(1,param_num) * 0.01
        b1 = np.random.rand(1,1) * 0.01
        W2 = np.random.rand(1,1) * 0.01
        b2 = np.random.rand(1) * 0.01        
        W = {'W1':W1,
             'W2':W2}
        b = {'b1':b1,
             'b2':b2}
        return W,b
    def propagate(self,w, b, X, Y = None,calculate_cost = True):
        m = X.shape[1]
        W1 = w['W1']
        b1 = b['b1']
        W2 = w['W2']
        b2 = b['b2']
        Z1 = np.dot(W1,X)+b1 
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = 1/(1+np.exp(-Z2))
        if calculate_cost:
            cost = 0
            cost = -1/m*(Y*np.log(A2)+(1-Y)*np.log(1-A2)).sum()
            cost = np.squeeze(cost)
            
            cache = {'Z1':Z1,
                     'A1':A1,
                     'Z2':Z2,
                     'A2':A2}
            return cache,cost
        else:
            return A2
    def backward(sefl,cache,w,b,X,Y,lambd):
        W1 = w['W1']
        b1 = b['b1']
        W2 = w['W2']
        b2 = b['b2']
        
        Z1 = cache['Z1']
        A1 = cache['A1']
        Z2 = cache['Z2']
        A2 = cache['A2']
        
        m = A2.shape[1]
        
        dZ2 = A2 - Y
        dW2 = 1/m*np.dot(dZ2,A1.T) + lambd/m*W2
        db2 = 1/m*np.sum(dZ2,axis = 1,keepdims = True)
        dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
        dW1 = 1/m*np.dot(dZ1,X.T) + lambd/m*W1
        db1 = 1/m*np.sum(dZ1,axis = 1,keepdims = True)

        grads = {"dW2": dW2,
                 "db2": db2,
                 "dW1": dW1,
                 'db1': db1}
        return grads
    
    def optimize(self,w, b, X, Y, num_iterations,recNum,batch_size, learning_rate,lambd):
        costs = []
        bith_numes = int(recNum/batch_size)
        bith_numes1 = bith_numes - 1
        for i in range(num_iterations):
            for j in range (bith_numes):
                if j == bith_numes1 :
                    X1 = X[:,j*batch_size:]
                    Y1 = Y[:,j*batch_size:]
                else:
                    X1 = X[:,j*batch_size:(j+1)*batch_size]
                    Y1 = Y[:,j*batch_size:(j+1)*batch_size]

                cache, cost = self.propagate(w, b, X1, Y1)
                #输出训练正确率
                costs.append(cost)
                if i % 1 == 0 and j == bith_numes1:
                    print ("Cost after iteration %i: %f" %(i, cost))
                    trainaccuracy = (cache['A2']+0.5).astype(np.int)
                    trainaccuracy = np.sum(trainaccuracy == Y1) / trainaccuracy.shape[1] * 100
                    print('Train_accuracy:%.2f' %trainaccuracy)
                grads = self.backward(cache,w,b,X1,Y1,lambd)
                
                dW2 = grads["dW2"]
                db2 = grads["db2"]
                dW1 = grads["dW1"]
                db1 = grads["db1"]
                w['W1'] = w['W1']-learning_rate*dW1
                b['b1'] = b['b1']-learning_rate*db1
                w['W2'] = w['W2']-learning_rate*dW2
                b['b2'] = b['b2']-learning_rate*db2 
        return w,b
    def error_rate(self, recNum, label, preval):
        return np.power(label - preval, 2).sum()

    def predict(self):
        self.loadTestData()
        preval = self.propagate(self.Weight,self.base,((self.feats_test-self.mean)/(self.var)).T,calculate_cost = False)
        preval = preval.T
        self.labels_predict = (preval+0.5).astype(int)
        self.savePredictResult()
    def train(self):
        self.loadTrainData()
        recNum = self.feats.shape[1]
        param_num = self.feats.shape[0]
        W,b =self.initParams(param_num)
        W,b =self.optimize(W, b,self.feats, self.labels, 6,recNum,512, 0.7,90)
        self.Weight = W
        self.base = b
        
def print_help_and_exit():
    print("usage:python3 main.py train_data.txt test_data.txt predict.txt [debug]")
    sys.exit(-1)


def parse_args():
    debug = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'debug':
            print("test mode")
            debug = True
        else:
            print_help_and_exit()
    return debug


if __name__ == "__main__":
    
    tstar = time.process_time()    
    
    debug = parse_args()
    train_file =  "train_data.txt"
    test_file = "test_data.txt"
    predict_file = "result.txt"
    lr = LR(train_file, test_file, predict_file)
    lr.train()
    lr.predict()

    if True or debug:
        answer_file ="answer.txt"
        f_a = open(answer_file, 'r')
        f_p = open(predict_file, 'r')
        a = []
        p = []
        lines = f_a.readlines()
        for line in lines:
            a.append(int(float(line.strip())))
        f_a.close()

        lines = f_p.readlines()
        for line in lines:
            p.append(int(float(line.strip())))
        f_p.close()

        print("answer lines:%d" % (len(a)))
        print("predict lines:%d" % (len(p)))

        errline = 0
        for i in range(len(a)):
            if a[i] != p[i]:
                errline += 1

        accuracy = (len(a)-errline)/len(a)
        print("accuracy:%f" %(accuracy))
        tend = time.process_time()
        print("总运行时间: %s s" % (str(tend - tstar)))