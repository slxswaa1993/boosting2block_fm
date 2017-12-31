import numpy as np
from metrics import *
from sklearn.utils.extmath import safe_sparse_dot
import scipy.sparse as sp



def loadB2bModel(file_name):
  with open(file_name,"rb") as fo:
    W=pickle.load(fo)
    Z=pickle.load(fo)

  return (W,Z)


def loadKerasModel():
    fo=open("../../data/keras_model.pkl","rb")
    user_factors=pickle.load(fo)
    item_factors=pickle.load(fo)
    return (user_factors,item_factors)


def loadPrfmModel():
  with open("../../data/prfm_weight_epoc_-1.pkl","rb") as fo:
    W=pickle.load(fo)
    Z=pickle.load(fo)
  return (W,Z)

if __name__=="__main__":

    # weight_file="/Users/dong/Desktop/BoostingFM-IJCAI18/dataset/ml-100k/12_17_20_51/models/u1_iter_63_save_weight.pkl"
    # # # 0.1582
    # W,Z=loadB2bModel(weight_file)
    # all_map=predic_map_FM_like(W, Z)
    # print 'b2b',all_map

    # 0.1831
    prfm_W, prfm_Z = loadPrfmModel()
    all_map_prfm=predic_map_FM_like(prfm_W, prfm_Z)
    print 'prfm',all_map_prfm

    # user_factors, item_factors = loadKerasModel()
    # all_map = predic_map_bpr(user_factors, item_factors, [1,3,5])
    # print all_map
    # print "k={},all_map={0}".format(all_map)

