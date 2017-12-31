# -*-coding:utf8-*-

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback
import gc
import math

class QuadraticSolver():

    def __init__(self):
        pass
       
    @staticmethod
    def getEigenvector(A):
        '''
        :param A: the matrix to be decomposed，A should be symmetric
        :return:
        '''
        # todo: use power method
        eigenval, eigenvec = eigsh(A, k=1)
        return eigenval, eigenvec
    
    @staticmethod
    def getComponentZ_eigval(context_num, U, s_dim, X_uv, X_uf):
        '''
        :param s_dim: 输入向量的维度
        return: vv^T、eigenval
        ''' 
        P= X_uv.multiply(U)
        Q= X_uf.multiply(U)
        
        A = safe_sparse_dot(P.T,X_uv) - safe_sparse_dot(Q.T,X_uf) 
        #print 'A.shape,type',A.shape,type(A)
        eigenval, eigenvec = QuadraticSolver.getEigenvector(A)
        #     print eigenvec.shape
        return safe_sparse_dot(eigenvec, eigenvec.T), eigenval
    
    
    @staticmethod
    def get_Hc(Ac, z_t):
        '''
        Note：Z_t 是一个对称矩阵
        '''
        return np.sum(np.diag(safe_sparse_dot(Ac, z_t)))
    #######################################################
    
    @staticmethod
    def getG_ration(W, W_old,P,Q):
        '''
        :return G matrix (N,1)
        '''

        # 这里为了验证去掉线性损失比，对于特征值分解的影响。
        num=P.shape[0]
        return np.mat(np.array([1.]*num).reshape(num,1)) 
        
        ########下面为正常代码######
        temp=P-Q
        current=np.exp(-temp.dot(W).todense())
        old=np.exp(-temp.dot(W_old).todense())       
        G=current/old
        
        print 'G_ration:',G.shape,type(G),'mean:',np.mean(G),'min',np.min(G),'max',np.max(G)
        del current
        del old
        return G
        ########以上为正常代码######
        
        # 这里为了验证去掉线性损失比的分母，对于特征值分解的影响。
        temp=P-Q    
        current=np.exp(-temp.dot(W).todense())       
        return current
        
 
    
    @staticmethod
    def getDia(Z,X):
        '''
        :注意,这里的Z有问题  必须是 numpy.matrixlib.defmatrix.matrix'
        :return (XZX.T).diagonal()
        '''

        H_P_diag_syn=None
        batch_size=10000
        total_sample=X.shape[0]
        if total_sample < batch_size:
            start=0
            end=total_sample
            batch_numbers=1
        else:
            start=0
            end=batch_size
            batch_numbers = int(math.ceil((total_sample*1.)/batch_size))

        batch_diag_list=[]
        for i in range(batch_numbers):
            H_P_batch=safe_sparse_dot(safe_sparse_dot(X[start:end],Z),X[start:end].T)
            dia_batch=H_P_batch.diagonal()
            start=end
            end +=batch_size
            if end > total_sample:
                end = total_sample
 
            batch_diag_list.append(dia_batch)   
        
        H_P_diag_syn = np.concatenate((batch_diag_list))

        return H_P_diag_syn
        
    @staticmethod
    def getH(Z,P,Q):
    
        '''
        这里会遇到严重的内存问题，当8w条数据参与计算的时候，会产生(8W，8W)的矩阵，这是会导致内存溢出
        :return H matrix (N,1)
        '''
        H_P_dia=QuadraticSolver.getDia(Z,P)
        H_Q_dia=QuadraticSolver.getDia(Z,Q)

        H=H_P_dia-H_Q_dia
       
        rst=np.asmatrix(H).T
        
        # H 为列向量
        return rst
    
    
    
    @staticmethod
    def getU_J(lambda_t,U,G,H):
        '''
        需要对U进行放大，使得所有的U加起来的等于样本总数。
        U=G*U*exp(-lambda_t*H)
        在更新U的时候需要用到【线性项损失比】,而线性项使用带用户信息的数据时，在计算损失的时候，也要带上用户信息
        G 是线性相比例不变，这里暂时不使用
        返回的U 是一个列向量
        numpy.matrixlib.defmatrix.matrix' (58196, 1)
        '''
        try:
            H_exp=np.exp(lambda_t*H)
        except AttributeError:
            fo=open("ExceptionU_J.pkl",'wb')
            pickle.dump(lambda_t,fo)
            pickle.dump(H,fo)
            pickle.dump(U,fo)
            fo.close()
            raise Exception("异常！！！")
            
            
        L=1.0/U-1
        M=np.multiply(L,H_exp)
        F=M+1
        U_J=1.0/F
        
        #U_J=U_J/np.sum(U_J)
        return U_J
    

    @staticmethod
    def costfun_lambda_t(H,a_3,U,lambda_t,G): 
            '''
             本部分对应p.s.m learning with boosting 公式(8)
             :prarm H matrix (N,1)      should be ndarray,has shape (N,)
             :prarm U matrix (N,1)     should be list,has shape (N,)
             :param G matrix (N,1)
              
            注意U的数据类型。
            <type 'numpy.ndarray'>   H
            <class 'numpy.matrixlib.defmatrix.matrix'> G
            <type 'list'> U 注意U的数据类型。一次迭代后可能会变
            <type 'numpy.ndarray'> exp
             
            F=(H-a_3)*G*(U)*(exp) 这里全部是点乘 
            '''
            U_J=QuadraticSolver.getU_J(lambda_t,U,G,H)
            cost_lambda_t=np.dot(H.T,U_J)-a_3        
            return cost_lambda_t

    @staticmethod
    #def lambdaSearch(context_num, z_t, U, a_3, lambda_epsilon, inital_interval, W, W_old, X_uv, X_uf,C):
    def lambdaSearch(context_num, z_t, U, a_3, lambda_epsilon, inital_interval,G,H):      
        
        def bi_search(lambda_l, lambda_u, epsilon):
            lambda_mid = 0.0
            search_times=0
            while True:
                lambda_mid = 0.5 * (lambda_l + lambda_u)
                start=time.time()                                
                cost_lambda_t = QuadraticSolver.costfun_lambda_t(H,a_3,U,lambda_mid,G)
                if cost_lambda_t > 0:
                    lambda_l = lambda_mid
                else:
                    lambda_u = lambda_mid
                if lambda_u - lambda_l < epsilon:  # 这里要小心，太小的浮点运算可能不准确。 
                    break
                search_times+=1       
            return lambda_mid,search_times
        
        start=time.time()
        lambda_l=QuadraticSolver.costfun_lambda_t(H,a_3,U,0,G)
      
        # find a proper upbound
        up=2
        lambda_u=QuadraticSolver.costfun_lambda_t(H,a_3,U,up,G)
        while  lambda_u > 0:
            up*=2
            lambda_u=QuadraticSolver.costfun_lambda_t(H,a_3,U,up,G)
        
        # 计算 lambda_t=0 的 costfun，如果小于零，则直接结束训练
        if lambda_l < 0:
            return -1,-1
        #print '每一次costfun_lambda_t','耗时:',(time.time()-start)/120,'min'
        #print 'lambda_t=',0,'costfun_lambda_t=',lambda_l
        #print 'lambda_t=',up,'costfun_lambda_t=',lambda_u
        #print '########################'
        
        return bi_search(0, up, lambda_epsilon)
        

    @staticmethod
    def get_G_H(W, W_old, z_t,X_uv, X_uf):

        G=QuadraticSolver.getG_ration(W, W_old, X_uv, X_uf)

        H=QuadraticSolver.getH(z_t,X_uv,X_uf)
        return (G,H)


    #######################################################
    @staticmethod
    def updateU(lambda_t,U,G,H):
        '''
            需要对U进行放大，使得所有的U加起来的等于样本总数。
            U=G*U*exp(-lambda_t*H)
            在更新U的时候需要用到【线性项损失比】,而线性项使用带用户信息的数据时，在计算损失的时候，也要带上用户信息
        '''        
        U=QuadraticSolver.getU_J(lambda_t,U,G,H)
        return U


        

