# -*-coding:utf8-*-
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import sys
import time


def get_non_interact_items(userID, train_data, test_data, min_tagID, max_tagID):
    # oberved items
    p_train_items_set = set(train_data[train_data['userID'] == userID]['itemID'].values)
    p_test_items_set = set(test_data[test_data['userID'] == userID]['itemID'].values)

    complete_p_items_set = p_train_items_set | p_test_items_set
    # 从所有负标签中随机的取出一定数目负标签
    c_negative_items = []

    cadidte=set(range(max_tagID+1))

    tartget=cadidte-complete_p_items_set

    for itemID in tartget:
        c_negative_items.append(itemID)

    np.random.shuffle(c_negative_items)
    return c_negative_items


def getSytheticTrainData(train_data, user_set, test_data, min_tagID, max_tagID):
    '''
    合成数据是为每个样本生成一个负样本
    '''
    sythetic_data = train_data
    sythetic_data['negItemID'] = 0
    for userID in user_set:
        c_neg_items = get_non_interact_items(userID, train_data, test_data, min_tagID, max_tagID)
        # 在原始数据中去除包含userID的所有的记录的index
        contex_indecise = sythetic_data[(sythetic_data['userID'] == userID)].index.values
        count = 0
        total_neg=len(c_neg_items)
        for index in contex_indecise:
            sythetic_data.loc[index]['negItemID'] = c_neg_items[count%total_neg]
            count += 1
    return sythetic_data


def create_synthentic_data_jester_2():
    datapath = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/jester-2/'
    train_file = datapath + "b2b_train_50_5_data.dat.txt"
    train_data = pd.read_table(names=['userID', 'itemID', 'rating'], filepath_or_buffer=train_file, )
    del train_data["rating"]
    MAX_USER_ID = int(train_data.describe()['userID'].loc["max"])
    MAX_ITEM_ID = int(train_data.describe()['itemID'].loc["max"])
    train_user_set = np.unique(train_data['userID'].values)
    # total_data.describe()
    # print type(user_set),len(user_set)

    test_file = datapath + "b2b_test_50_5_data.dat.txt"
    test_data = pd.read_table(names=['userID', 'itemID', 'rating'], filepath_or_buffer=test_file, )
    test_user_set = np.unique(test_data['userID'].values)
    del test_data["rating"]
    user_set = set(train_user_set) | set(test_user_set)

    synsdata = getSytheticTrainData(train_data, user_set, test_data, 0, MAX_ITEM_ID)

    synsdata.to_csv(datapath+'jester-2_syntheic_data.csv',header=False,index=False,sep='|')

    print 'Done!'


if __name__ == "__main__":
    create_synthentic_data_jester_2()