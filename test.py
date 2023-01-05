'''
test
'''
import os
import pandas as pd
import numpy as np

import util
from option import args
from data_prepare import Test_data_prepare,Train_data_prepare
from tester import Tester


class Test():
    def __init__(self):
        self.test_target_len= 120
        self.predict_column='Sunspots'

    def set(self,*,test_target_len,predict_column):
        self.test_target_len= test_target_len
        self.predict_column =  predict_column

    def run(self):
        path_result = args.path_result
        file_name ='result.npy'


        path_min_max = os.path.join(args.dir_data,"min_max")
        filename_min_max = 'train_min_max.txt'

        #test data_prepare
        test_data_pre = Test_data_prepare(args)
        test_data_pre.set(test_target_len =self.test_target_len,predict_column=self.predict_column)
        test_data_pre.prepare()

        #test
        tester = Tester(args)
        tester.set(test_target_len=self.test_target_len)
        tester.test()

        # load series from file
        predict_series = np.load(os.path.join(path_result,file_name))
        # use instantiated normalizer to recover series to original value
        dic = util.load_variable(os.path.join(path_min_max,filename_min_max))
        nomalizer = util.Normalize()
        nomalizer.set_scaler(dic['min'],dic['max'])
        predict_series = nomalizer.inverse_transform(predict_series).reshape(-1)

        # save result to file
        data = pd.read_csv(os.path.join(args.path_result,'result.csv'))
        for i in range(0,len(predict_series)):
            data.loc[len(data)+i,args.predict_column] = predict_series[i]
        data.fillna('NaN')
        data.to_csv(os.path.join(args.path_result,'result.csv'),index = 0,float_format='%.2f')

if __name__ == "__main__":
    test = Test()
    test.run()