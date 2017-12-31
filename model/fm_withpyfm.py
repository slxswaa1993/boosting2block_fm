# coding:utf8
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
import random
from scipy.io import mmread
import scipy.sparse as sp
import sys
from  eval.metrics import *

if __name__=="__main__":
    datapath = '/home/zju/dgl/dataset/recommend/ml-100k/'
    train_data_file = datapath + 'train_point_wise.pkl'
    import pickle

    with open(train_data_file) as fo:
        train_X = pickle.load(fo)
        train_y = pickle.load(fo)

    # 当做分类问题
    fm = pylibfm.FM(num_factors=50, num_iter=100, verbose=True, task="classification", initial_learning_rate=0.0001,
                    learning_rate_schedule="optimal")
    fm.fit(train_X_64, train_y)
    all_map = predic_map_FM_like(fm, [1, 3, 5])
    print 'fm', all_map