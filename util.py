'''
this file provided some useful tools.
1. Functions for variable save and load. eg.min, max value to recover normalized data.
    save_variable(v, filename),
    load_variable(filename)
2. Function for deviding time series into  window_sized train and validation sub series.
    devide_series(series,train_source_len,train_target_len,validation_source_len,validation_target_len, stride):
3. Normalize Class. Provide functions that map data to proper range eg.(0,1) and map back。
    set_scaler(self,min,max),
    transform(self,X),
    inverse_transform(self,X_fited).
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle



def save_variable(v, filename):
    '''
    save variable into file
    :param v: variable to be saved
    :param filename: save path
    :return:
    '''
    f = open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return

def load_variable(filename):
    '''
    load variable from file
    :param filename: save path
    :return r:  saved variable
    '''
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def get_test_series(series,test_source_len):
    '''
        obtain test series from the tail.
        '''
    test_series = series[-test_source_len:]
    return test_series

def get_train_series(series,train_source_len,train_target_len,stride):
    window_len = train_source_len + train_target_len

    # obtain train series
    remain_len = (series.size - window_len) % stride
    train_series = np.empty((0, window_len))
    for pos in range(remain_len, series.size - window_len + 1, stride):
        temp = series[pos:pos + window_len]
        train_series = np.row_stack((train_series, temp))

    train_series_s = train_series[..., :train_source_len]
    train_series_t = train_series[..., train_source_len:]

    return train_series_s,train_series_t


def get_validation_series(series,source_len,target_len):
    '''
    obtain validation series from the tail.
    '''
    validation_series_s = series[-target_len - source_len:-target_len]
    validation_series_t = series[-target_len:]
    if target_len == 0:
        validation_series_s = series[-target_len - source_len:]
    validation_series_t = validation_series_t.reshape(1,-1)
    validation_series_s = validation_series_s.reshape(1,-1)
    return validation_series_s, validation_series_t


def devide_series(series,train_source_len,train_target_len,validation_source_len,validation_target_len, stride):
    '''
    obtain train and validation slide window series from 1d time series.

    :param series: 1 d numpy array
    :param train_source_len: window length of series for input image
    :param train_target_len: window length of series for  output image
    :param validation_target_len: window length of series for input image
    :param validation_source_len: window length of series for  output image
    :param stride: widow stride
    :return: train_series_s (window_num,train_source_len), train_series_t(window_num,train_target_len),
            validation_series_s(validation_source_len), validation_series_t(validation_target_len)
    '''
    window_len = train_source_len+train_target_len

    #obtain validation series from the tail
    validation_series_s, validation_series_t = get_validation_series(series,validation_source_len,validation_target_len)

    #obtain train series
    series = series[:-validation_target_len]
    remain_len = (series.size - window_len) % stride
    train_series = np.empty((0, window_len))
    for pos in range(remain_len, series.size - window_len+1, stride):
        temp = series[pos:pos+window_len]
        train_series = np.row_stack((train_series,temp))

    train_series_s = train_series[...,:train_source_len]
    train_series_t = train_series[...,train_source_len:]
    # print(train_series_s.shape)

    return train_series_s, train_series_t,validation_series_s, validation_series_t

class Normalize:
    def __init__(self,samplerange=(1e-8,1-1e-8)):
        self.samplerange = samplerange
        self.scaler = None

    def set_scaler(self,min,max):
        self.scaler = MinMaxScaler(self.samplerange)
        self.scaler.fit([[min],[max]])

    def transform(self,X):
        '''
        Transform data series to samplerange. Need set scaler first.set_scaler(self,min,max)
        :param X: numpy array
        '''
        X_fited = self.scaler.transform(X.reshape(-1,1)).reshape(X.shape)

        return X_fited

    def fit_transform(self, X):
        '''
        Transform data series to samplerange. Fit data automatically and don't need set scaler.
        :param X: numpy array
        '''
        min, max = np.min(X), np.max(X)
        self.scaler = MinMaxScaler(self.samplerange)
        X_fited = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)

        return X_fited, min, max

    def inverse_transform(self,X_fited):
        '''
        Convert fited data series to original.Need set scaler first.set_scaler(self,min,max)
        :param X_fited: fitted numpy array, shape=(n_samples,n_timestamps)
        :return X: numpy array, shape=(n_samples,n_timestamps)
        '''
        if self.scaler is None:
            raise Exception("self.scaler is None")
        X = self.scaler.inverse_transform(X_fited.reshape(-1, 1)).reshape(X_fited.shape)

        return X

if __name__ == "__main__":
    pass