# -*-coding:utf8-*-
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import sys
import time


def getOneHot(ID,MAX_ID):
    '''
    ID 和 MAX_ID 都是0索引
    return: ndarray, (MAX_ID+1,1), e.g [0.0,...,1.0,...,0.0]
    '''
    onehot=np.array([0.0]*(MAX_ID+1))
    onehot[ID]=1.0
    return onehot

def createX_ciX_cj(sythetic_data, MAX_USER_ID=701, MAX_ITEM_ID=1521):
    X_ci = None
    X_cj = None
    total_num = sythetic_data.shape[0]
    for index in sythetic_data.index:
        userID = sythetic_data.loc[index]['userID']
        movieID = sythetic_data.loc[index]['itemID']
        neg_movieID = sythetic_data.loc[index]['negItemID']

        oht_userID = sp.csc_matrix(getOneHot(userID, MAX_USER_ID), dtype='f')
        oht_movieID = sp.csc_matrix(getOneHot(movieID, MAX_ITEM_ID), dtype='f')
        oht_neg_ItemID = sp.csc_matrix(getOneHot(neg_movieID, MAX_ITEM_ID), dtype='f')

        x_ci = sp.hstack([oht_userID, oht_movieID])
        if X_ci == None:
            X_ci = x_ci
        else:
            X_ci = sp.vstack([X_ci, x_ci])

        ### 负样本生成

        # 生成x_cj
        x_cj = sp.hstack([oht_userID, oht_neg_ItemID])
        if X_cj == None:
            X_cj = x_cj
        else:
            X_cj = sp.vstack([X_cj, x_cj])

        #         if index > 100:
        #              break

        sys.stdout.write("\rHandling context %d of %d" % (index + 1, total_num))
        sys.stdout.flush()
    return X_ci, X_cj


if __name__=='__main__':

    datapath = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/jester-2/'
    train_file=datapath + 'jester-2_syntheic_data.csv'
    df_syn_data = pd.read_csv(train_file, sep='|', names=['userID', 'itemID', 'negItemID'])
    X_ci, X_cj = createX_ciX_cj(df_syn_data,MAX_USER_ID=701,MAX_ITEM_ID=145)
    # print ''
    with open(datapath+"jester-2_X_ci_X_cj.pkl", 'wb') as fo:
        pickle.dump(X_ci, fo)
        pickle.dump(X_cj, fo)