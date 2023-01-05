'''
train and test.
'''
from option import args
from data_prepare import Test_data_prepare,Train_data_prepare
from trainer import Trainer

class Train():
    def run(self):
        #train data_prepare
        train_data_pre = Train_data_prepare(args)
        train_data_pre.prepare()

        # train
        trainer = Trainer(args)
        trainer.train()



