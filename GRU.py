import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import GRU
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#数据手动处理 预测数据另成一列 上移一行 删除最后一行 形成temp.csv文件
scaler = MinMaxScaler(feature_range=(0, 1))

##生成训练和测试数据


def generate_model_data(data_path,alpha,days):
    #data_path:要读取的csv文件
    #alpha:训练数据比例  0.8=80%
    #days:时间步
    df =pd.read_csv(data_path)
    train_day=int(len(df['close'])-days+1)#701-15+1=687总共的样本
    for property in ['open','close','high','low','volume','next_close']:
        df[property]=scaler.fit_transform(np.reshape(np.array(df[property]),(-1,1)))
    X_data,Y_data=list(),list()
    #生成时序数据
    for i in range(train_day):
        Y_data.append(df['next_close'][i+days-1])
        for j in range(days):
            for m in ['open','close','high','low','volume']:
                X_data.append(df[m][i+j])

    X_data=np.reshape(np.array(X_data),(-1,5*days))#二维数组 687个5*15
    train_length=int(train_day*alpha)#687*0.8=549.6====549个训练数据

    X_train=np.reshape(np.array(X_data[:train_length]),(len(X_data[:train_length]),days,5))#三维数组
    X_test=np.reshape(np.array(X_data[train_length:]),(len(X_data[train_length:]),days,5))
    Y_train,Y_test=np.array(Y_data[:train_length]),np.array(Y_data[train_length:])
    print(Y_test.shape )
    return X_train,Y_train,X_test,Y_test

#MAPE
def calc_MAPE(real,predict):
    Score_MAPE=0
    for i in range(len(predict[:,0])):
        Score_MAPE+=abs((real[:,0][i]-predict[:,0][i])/real[:,0][i])
    Score_MAPE=Score_MAPE/len(predict[:,0])
    return Score_MAPE



def calc_AMAPE(real,predict):
    Score_AMAPE=0
    Score_AMAPE_DIV=sum(real[:,0])/len(real[:,0])#真实值的均值
    for i in range(len(predict[:,0])):
        Score_AMAPE +=abs((real[:,0][i]-predict[:,0][i])/Score_AMAPE_DIV)
    Score_AMAPE=Score_AMAPE /len(predict[:,0])
    return Score_AMAPE


def evaluate(real,predict):
    RMSE=math.sqrt(mean_squared_error(real[:,0],predict[:,0]))
    MAE=mean_absolute_error(real[:,0],predict[:,0])
    MAPE=calc_MAPE(real,predict)
    AMAPE=calc_AMAPE(real,predict)
    return RMSE,MAE,MAPE,AMAPE


def GRU_model(X_train,Y_train,X_test,Y_test):
    #X_train.shape:687 15 5
    model=Sequential()
    model.add(GRU(units=20,input_shape=(X_train.shape[1],X_train.shape[2])))

    model.add(Dense(1,input_dim=(15,20),activation='hard_sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,Y_train,epochs=200,batch_size=20,verbose=1)#进行训练

    trainPredict=model.predict(X_train)#返回预测值trainPredict
    trainPredict=scaler.inverse_transform(trainPredict)#将标准化的数据转换为原来的比例
    Y_train=scaler.inverse_transform(np.reshape(Y_train,(-1,1)))

    testPredict=model.predict(X_test)
    testPredict=scaler.inverse_transform(testPredict)
    Y_test=scaler.inverse_transform(np.reshape(Y_test,(-1,1)))
    return Y_train,trainPredict,Y_test,testPredict


if __name__=='__main__':#都是两个杠
    alpha=0.8
    days=15

    X_train,Y_train,X_test,Y_test=generate_model_data('temp.csv',alpha,days)

    train_Y,trainPredict,test_Y,testPredict=GRU_model(X_train,Y_train,X_test,Y_test)

    #画图
    plt.plot(list(trainPredict),color='red',label='prediction')
    plt.plot(list(train_Y),color='blue',label='real')
    plt.legend(loc='upper left')#图例位置
    plt.title('train data')
    plt.show()

    plt.plot(list(testPredict),color='red',label='prediction')
    plt.plot(list(test_Y),color='blue',label='real')
    plt.legend(loc='upper left')
    plt.title('test data')
    plt.show()

    #
    RMSE,MAE,MAPE,AMAPE=evaluate(test_Y,testPredict)
    print(RMSE,MAE,MAPE,AMAPE)











