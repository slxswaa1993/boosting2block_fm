# -*——coding:utf8-*-
from quadratic_solver_matrix import *
#import quadratic_solver_matrix_with_user as qc
#from linear_solver import *
import linear_solver_with_user as lsu
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback
import sys
import datetime
import os
from functions import *

def init_U():
    '''
     U 的初始权重分配：
         1. 以用户为单位进行分配
         2. 观测样本少的用户，权重大
         3. 权重需要进行统一放大
    '''
    
    datapath_bpr='/home/zju/dgl/source/baseline/models/adabpr_code/datasets/'
    train_file=datapath_bpr+'ml_100k_occf_training.txt'
    test_file=datapath_bpr+'ml_100k_occf_testing.txt'
    train_data, Tr, Tr_neg, Te = data_process(train_file, test_file)
    D0 = np.array([1.0/float(Tr[u]['num']) for u, i in train_data])
    D = (D0/np.sum(D0))*train_data.shape[0]
    return D

def initial(context_num,d_dim):
    U=init_U()
    U=np.asmatrix(U).T
    ## W_old 设为0
    W_old=sp.csr_matrix(np.random.uniform(low=0./d_dim, high=0./d_dim, size=d_dim).reshape(d_dim,1))
    ## W_old 设为单位矩阵
    Z=sp.csr_matrix((d_dim,d_dim),dtype='f')
    #Z=sp.identity(d_dim, dtype='f', format="csr")
    return U,W_old,Z

def save2pkl(iter_count,flod_name,datapath,Z,W):
    print 'saving model...'
    print 'modelpath',datapath
    fo=open(datapath+flod_name+'_iter_'+str(iter_count)+'_save_weight.pkl','wb')
    pickle.dump(W,fo)
    pickle.dump(Z, fo)
    fo.close()
    print 'saving model end'
    
def getLinearElementLoss_with_user(W,X_ci,X_cj,C):
    '''
    :param W should be a column vector
    :param X_ci should be sparse matrix
    :param C 用户信息
    return exp(-W.Tb_i) is (80000,) ndarray
    '''
    B=X_ci-X_cj+C    
    linear_loss=(safe_sparse_dot(B,W))
    temp=np.exp(-linear_loss.data)
    return temp


def getLinearElementLoss(W,X_ci,X_cj):
    '''
    :param W should be a column vector
    :param X_ci should be sparse matrix
    :param C 用户信息
    return exp(-W.Tb_i) is (80000,) ndarray
    '''
    B=X_ci-X_cj   
    linear_loss=(safe_sparse_dot(B,W))
    temp=np.exp(-linear_loss.data)
    return temp


def getQuadraticLoss_with_user(Z,X_ci,X_cj,C,linearSolver):
    print 'getQuadraticLoss',X_ci.shape
    
    PAI=linearSolver.getPAI(X_ci,X_cj,Z,C)
    return PAI

def getQuadraticLoss(Z,X_ci,X_cj,linearSolver):
    print 'getQuadraticLoss',X_ci.shape
    
    PAI=linearSolver.getPAI(X_ci,X_cj,Z)
    return PAI

def getTotalObjLoss(ele_linear_loss,ele_quadratic_loss,a_1,a_3,W,lambda_list):
    '''
    :param ele_linear_loss is a one-dim ndarray
    :param ele_quadratic_loss is a one-dim ndarray
    '''
    assert len(ele_linear_loss)==len(ele_quadratic_loss)
    size=len(ele_linear_loss)
    loss=np.sum(ele_linear_loss*ele_quadratic_loss)/size
    linear_regular=0.5*a_1*safe_sparse_dot(W.T,W)
    ## 算出来也是csc_matrix，所以要转一下
    linear_regular=linear_regular.data[0]
    if len(lambda_list)!=0:
        qudratic_regular=a_3*np.sum(lambda_list)
    else:
        qudratic_regular=0.
    print 'regular:W=',linear_regular,'Z=',qudratic_regular,'pure loss',loss    
    return loss+linear_regular+qudratic_regular

def getPureContext(X_ci,context_num):
    pure_C=X_ci[:,:943]
    padding=sp.csr_matrix((context_num,1522),dtype='f')
    C=sp.hstack([pure_C,padding])
    C=sp.csr_matrix(C)
    return C

def train(boosting_iters,X_uv,X_uf,linear_epoc,batch_size,eta,a_1,a_3,lambda_epsilon,context_num,save_model):
    '''
    :param boosting_iters
    :param X_uv context-observed 数据,csr_matrix 矩阵，每一行一个样本
    :param X_uf context-non_observed 数据,csr_matrix 矩阵，每一行一个样本
    :param linear_epoc 计算线性项的时候,迭代的周期数
    :param batch_size  计算线性项的时候,batch 的大小
    :param a_1 线性正则项参数
    :param a_3 二次项正则参数
    :param eta 这指的是在使用梯度下降计算线性项的时候，步长的大小。（随后改进：这个参数应该随训练的进行而减小）
    :param lambda_epsilon 二次项超参数，控制lambda_t 的精度 e.g 0.001,0.01
    :param context_num  这里的context_指的是在计算Z的时候，使用的样本总数，与线性项计算无关，与真正的上下文无关
    '''
    d_dim=X_uv.shape[1]
    U,W_old,Z=initial(context_num,d_dim)
    
    # 用户信息 
    C=getPureContext(X_uv,context_num)
    print 'C：',C.shape,type(C)
    
    # print 'U shape 17 train',U.shape,type(U)
    # create a folder to save the weight
    modelPath=str(datetime.datetime.now().month)+'_'+str(datetime.datetime.now().day)+'_'
    modelPath+=str(datetime.datetime.now().hour)+'_'+str(datetime.datetime.now().minute)
    datapath='/home/zju/dgl/dataset/recommend/ml-100k/'
    modelPath=datapath+modelPath+'/'
    
    if not os.path.exists(modelPath)and save_model:
        os.makedirs(modelPath)
    
   
    old_ele_quadratic_loss=[1]*context_num
    lambda_list=[]
    for iter_count in range(boosting_iters):
        print '##############################boosting_iter:',iter_count,'#########################'
        
        
#        print 'U[3457],sum(U):',U[3457],np.sum(U)
        ###
#         print 'context_num',context_num
        ###
        
        #linearSolver=LinearSolver(batch_size,linear_epoc,X_uv,X_uf,Z,a_1,eta)
        start=time.time()
        linearSolver=lsu.LinearSolver(batch_size,linear_epoc,X_uv,X_uf,Z,a_1,eta,C)
        W=linearSolver.fit()
        print 'W is finished:',W.shape,type(W),'耗时:',(time.time()-start)/60,'min'
        

        # 输出当前的loss
#         print '#######'
#         print 'W is finished:',W.shape,type(W),'耗时:',(time.time()-start)/60,'min'
        #start=time.time() 
    
        #ele_linear_loss=getLinearElementLoss(W,X_uv,X_uf)
        #ele_linear_loss=getLinearElementLoss_with_user(W,X_uv,X_uf,C)
        #total_loss=getTotalObjLoss(ele_linear_loss,old_ele_quadratic_loss,a_1,a_3,W,lambda_list)
        #print '##################### Loss A #####################'
        #print 'total_loss:',total_loss
        #print 'Linear loss:',total_loss,'耗时:',(time.time()-start)/60,'min'   
   
        
#         print '#######Update Z########'
        start=time.time()
        ## 这里的z_t是numpy.ndarray
        
        # quadratic_solver_matrix_with_user
        #z_t,eigenval=qc.QuadraticSolver.getComponentZ_eigval(context_num, U,d_dim,X_uv,X_uf,C)
        
        # quadratic_solver_matrix
        z_t,eigenval=QuadraticSolver.getComponentZ_eigval(context_num, U,d_dim,X_uv,X_uf)
        
        print 'eigenval is',eigenval,'耗时:',(time.time()-start)/60,'min'
#         print 'z_t',type(z_t)        

        # 这里仅做验证使用，真正使用的时候，应该去掉abs
        if eigenval < a_3:
            save2pkl(iter_count,'u1',modelPath,Z,W)
            break;       
            
            
                
        print '#####################模型权重更新#####################'
    
        start=time.time()   
        
        # quadratic_solver_matrix_with_user
        #lambda_t,search_times=qc.QuadraticSolver.lambdaSearch(context_num,z_t,U,a_3,lambda_epsilon,[0,10],W,W_old,X_uv,X_uf,C) 
        
        # quadratic_solver_matrix
        lambda_t,search_times=QuadraticSolver.lambdaSearch(context_num,z_t,U*X_uv.shape[0],a_3,lambda_epsilon,[0,10],W,W_old,X_uv,X_uf,C) 
        
        if lambda_t == search_times and search_times == -1:
            save2pkl(iter_count,'u1',modelPath,Z,W)
            break;
            
        
        print 'lambda_t is finished:',lambda_t,'耗时:',(time.time()-start)/60,'min','search_times:',search_times
        
   
        print '###################样本权重更新#####################'

        start=time.time()  
        # quadratic_solver_matrix_with_user
        #U=qc.QuadraticSolver.updateU(lambda_t,U,z_t,W, W_old, X_uv, X_uf,C)
        
        # quadratic_solver_matrix
        U=QuadraticSolver.updateU(lambda_t,U,z_t,W, W_old, X_uv, X_uf,C)
#         print 'update U finished:耗时:',(time.time()-start)/60,'min','search_times:'
        
        print "U要是概率分布:",np.sum(U)
        W_old=W
        # 此条件只在第一次迭代的时候成立
        if isinstance(Z,sp.csr.csr_matrix):
            assert iter_count==0
            Z=lambda_t*z_t
        else:
            Z+=lambda_t*z_t
            
#         print '合成的Z:',Z.shape,type(Z)
        
        
        lambda_list.append(lambda_t)
        
        # 输出当前的loss
        # 80000 条数据 算一遍需要 14.0426008503
#       start=time.time()        
        #ele_quadratic_loss=getQuadraticLoss(Z,X_uv,X_uf,linearSolver)
#        ele_quadratic_loss=getQuadraticLoss_with_user(Z,X_uv,X_uf,C,linearSolver)
#         print 'ele_quadratic_loss耗时:',(time.time()-start)/60,'min'
        
#        start=time.time()  
        #print 'ele_linear_loss:',ele_linear_loss.shape,'ele_quadratic_loss',ele_quadratic_loss.shape
#        total_loss=getTotalObjLoss(ele_linear_loss,ele_quadratic_loss,a_1,a_3,W,lambda_list)
#        print '##################### Loss B #####################'
#        print 'total_loss:',total_loss
          
            
        #保存当前的二次项loss,也就是线性项计算中的PAI，方便下一次计算线性项权重和loss    
#        old_ele_quadratic_loss=ele_quadratic_loss
        
        
        ######实验代码：动态更新a_3#######
        #a_3-=0.00002
        #print 'a_3减小',a_3
        #############
        
        # 节约存储空间，每隔5轮保存一次
        if save_model and iter_count%5==0:
            save2pkl(iter_count,'u1',modelPath,Z,W)

    return W,Z









