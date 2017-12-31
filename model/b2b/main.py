# -*——coding:utf8-*-
import pickle
#from boost2block import *
from boost2block_logistic import *
import time
import sys
import argparse

def load_data_file(train_data_file):
    '''
    从文件加载处理好的数据
    '''
    fi = open(train_data_file, 'rb')
    X_ci = pickle.load(fi)
    X_cj = pickle.load(fi)
    fi.close()
    X_ci = sp.csr_matrix(X_ci)
    X_cj = sp.csr_matrix(X_cj)
    return X_ci,X_cj


if __name__=='__main__':
    datapath='/home/zju/dgl/dataset/recommend/ml-100k/'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-flod_name', help='Training flod file. e.g u1', dest='flod_name', default='u1')   
    parser.add_argument('-iters', help='boosting iters', dest='iters', default=100,type=int)  
    parser.add_argument('-lepoc', help='linear epoc', dest='lepoc', default=20, type=int)   
    parser.add_argument('-batch-size', help='batch size when trian linear term', dest='batch_size', default=1000, type=int)     
    parser.add_argument('-eta', help='linear learning rate', dest='eta', default=0.01, type=float)
    parser.add_argument('-alpha1', help='hyper param of regulizer of linear term', dest='alpha_1', default=10, type=float)
    parser.add_argument('-alpha3', help='hyper param of regulizer of quadratic term', dest='alpha_3', default=0.001, type=float)   
    parser.add_argument('-debug', help='1,if true;else 0', dest='isDebug', default=-1, type=int) 
    parser.add_argument('-epsilon', help='precision of lambda ', dest='epsilon', default=0.0000001, type=float)
    parser.add_argument('-save', help='1,if true;else 0', dest='isSave', default=1, type=int) 
    args = parser.parse_args()
    
   
    print '############训练参数#############'
    
    print args
    
    print '############Begin#############'
    
#     flod_name=args.flod_name
    
    flod_name='bpr'
    #train_data_file = datapath+'bpr_orderd_short.pkl'
    train_data_file = datapath+'from_synthetic_data_csv.pkl'
    
    X_ci, X_cj=load_data_file(train_data_file)
    print "X_ci:",X_ci.shape
    context_num=X_ci.shape[0]
    if args.isDebug > 0:
        context_num = args.isDebug
    
    print 'context_num',context_num
    
    start=time.time()
    ## 注意的X_uv和X_uf的赋值
    
    ## 参数调试
    W,Z=train(boosting_iters=args.iters, 
                          X_uv=X_ci[:context_num],
                          X_uf=X_cj[:context_num],
                          linear_epoc=args.lepoc, 
                          batch_size=args.batch_size, 
                          eta=args.eta,
                          a_1=args.alpha_1, 
                          a_3=args.alpha_3, 
                          lambda_epsilon=args.epsilon, 
                          context_num=context_num,
                          save_model=args.isSave)
    
    print 'Training end,total time:',(time.time()-start)/60,'min'
    print 'Done!'
