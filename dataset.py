"""
Dataset class.
"""
import os
import numpy as np
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader

'''
Dataset for DataLoader
'''


class MyDataset(data.Dataset):
    def __init__(self, path_s, path_t, test=False):
        if test is False:
            filename_train_s = "train_s.npy"
            filename_train_t = "train_t.npy"
            self.sources = np.load(os.path.join(path_s, filename_train_s))
            self.targets = np.load(os.path.join(path_t, filename_train_t))
        else:
            filename_test_s = "test_s.npy"
            filename_test_t = "test_t.npy"
            self.sources = np.load(os.path.join(path_s, filename_test_s))
            self.targets = np.load(os.path.join(path_t, filename_test_t))

    def __getitem__(self, index):
        s = self.sources[index]
        s = s[np.newaxis, :]
        t = self.targets[index]
        t = t[np.newaxis, :]
        return s, t

    def __len__(self):
        return len(self.sources)


if __name__ == '__main__':

    train_base_dir = "data/elect/series"
    source_dir = train_base_dir
    target_dir = train_base_dir

    dataset = MyDataset(path_s=source_dir, path_t=target_dir, test=False)
    dataLoader = DataLoader(dataset, batch_size=5)

    i = 0
    for img, target in dataLoader:
        print("------{}------".format(i))
        print(img.size())
        print(target.size())
        i += 1
