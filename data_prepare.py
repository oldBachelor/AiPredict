'''
this file do some data preparing work.
1. divide them into train and test series
2. save series
3. translate them into GramianAngulaAddFiled images
4. save images
'''
import os.path
import pandas as pd
import util
# from gadf import GramianAngulaAddFiled
import numpy as np
from option import args

# datasets_base_dir = "data/mothly-sunsplots"
# path_series_seperate = os.path.join(datasets_base_dir,"series")
# path_images_seperate = os.path.join(datasets_base_dir,"images")
# path_min_max = os.path.join(datasets_base_dir,"min_max")
#
# #obtain datasets
# path_series_row = os.path.join(datasets_base_dir,"data_raw/monthly-sunspots.csv")
# data = pd.read_csv(path_series_row)
# series = data[1].values
#
# #create normalizer and map serie to (0,1)
# nomalizer = util.Normalize()
# series, min, max = nomalizer.fit_transform(series)
#
# # save min and max value to faciliate validation recover
# min_max = {"min":min,"max":max}
# util.save_variable(min_max,os.path.join(path_min_max,"min_max.txt"))
# # temp = util.load_variable(path_min_max)
#
#
# #devide series for train and validation
# train_sources_len = 270
# train_targets_len = 1
# test_sources_len = train_sources_len
# test_targets_len = 120
# stride = 1
# train_s, train_t, test_s, test_t = util.devide_series(series,train_sources_len,train_targets_len,test_sources_len, test_targets_len, stride)
#
# #save obtained series
# np.save(os.path.join(path_series_seperate,"train_s.npy"),train_s)
# np.save(os.path.join(path_series_seperate,"train_t.npy"), train_t)
# np.save(os.path.join(path_series_seperate,"test_s.npy"),test_s)
# np.save(os.path.join(path_series_seperate,"test_t.npy"),test_t)


# #transform series into pictures
# GADF = GramianAngulaAddFiled()
# train_s_img = GADF.seq2GAF(train_s)
# train_t_img = GADF.seq2GAF(train_t)
# test_s_img = GADF.seq2GAF(test_s)
# test_t_img = GADF.seq2GAF(test_t)

# #save obatained pictures
# np.save(os.path.join(path_images_seperate,"train_s.npy"),train_s_img)
# np.save(os.path.join(path_images_seperate,"train_t.npy"), train_t_img)
# np.save(os.path.join(path_images_seperate,"test_s.npy"),test_s_img)
# np.save(os.path.join(path_images_seperate,"test_t.npy"),test_t_img)

class Train_data_prepare():
    def __init__(self,args):
        self.dir_data = args.dir_data
        self.train_data = args.train_data_raw
        self.path_series_row = os.path.join(os.path.join(self.dir_data,'data_raw'), self.train_data)
        self.path_series_seperate = os.path.join(self.dir_data, "series")
        self.path_min_max = os.path.join(self.dir_data, "min_max")


        self.predict_column = args.predict_column
        self.train_source_len = args.train_source_len
        self.train_target_len = args.train_target_len
        self.stride = args.stride

    def prepare(self):
        # obtain datasets
        data = pd.read_csv(self.path_series_row)
        series = data[self.predict_column].values

        # create normalizer and map serie to (0,1)
        nomalizer = util.Normalize()
        series, min, max = nomalizer.fit_transform(series)

        # save min and max value to faciliate validation recover
        min_max = {"min": min, "max": max}
        util.save_variable(min_max, os.path.join(self.path_min_max, "train_min_max.txt"))

        # print(self.train_source_len)
        # print(self.train_target_len)
        # train_s, train_t, _,_ = util.devide_series(series, self.train_source_len, self.train_target_len,
        #                                                       0, 0, 1)
        train_s,train_t = util.get_train_series(series, self.train_source_len, self.train_target_len, self.stride)
        # save obtained series
        np.save(os.path.join(self.path_series_seperate, "train_s.npy"), train_s)
        np.save(os.path.join(self.path_series_seperate, "train_t.npy"), train_t)

class Test_data_prepare():
    def __init__(self, args):
        self.dir_data = args.dir_data
        self.test_data = args.test_data_raw
        self.path_series_row = os.path.join(os.path.join(self.dir_data,'data_raw'), self.test_data)
        self.path_series_seperate = os.path.join(self.dir_data, "series")
        self.path_min_max = os.path.join(self.dir_data, "min_max")

        self.path_result = args.path_result
        self.test_source_len = args.test_source_len

        self.test_target_len= 120
        self.predict_column='Sunspots'

    def set(self,*,test_target_len,predict_column):
        self.test_target_len= test_target_len
        self.predict_column =  predict_column

    def prepare(self):
        # obtain datasets
        data = pd.read_csv(self.path_series_row)
        series = data[self.predict_column].values

        #save data
        path_result = os.path.join(self.path_result,'result.csv')
        if os.path.exists(path_result):
            os.remove(path_result)
        data.to_csv(path_result,index = 0)

        # use train nomalizer min and max to map test data
        dic = util.load_variable(os.path.join(self.path_min_max, "train_min_max.txt"))
        nomalizer = util.Normalize()
        nomalizer.set_scaler(dic['min'], dic['max'])
        series = nomalizer.transform(series)

        test_s = util.get_test_series(series, self.test_source_len)

        path_series_seperate = os.path.join(self.path_series_seperate, "test_s.npy")
        if os.path.exists(path_series_seperate):
            os.remove(path_series_seperate)
        np.save(path_series_seperate, test_s)

if __name__ == "__main__":
    # train_data_prepare = Train_data_prepare(args)
    # train_data_prepare.prepare()

    test_data_prepare = Test_data_prepare(args)
    test_data_prepare.prepare()

