#coding: utf8
import copy
import math
from  eval.auc import *
from utils import data_path


def getXI(W, V, P, Q):
    '''
    return f(P)-f(Q)
    是一个列向量
    '''
    Z = np.dot(V, V.T)
    # p-q的线性项目
    start = time.time()
    Linear = (safe_sparse_dot((P - Q), W))

    # p-q的二次项目，暂不考虑去对角
    f_X_ci = safe_sparse_dot(safe_sparse_dot(P, Z), P.T)
    f_X_cj = safe_sparse_dot(safe_sparse_dot(Q, Z), Q.T)
    Qudratic = (f_X_ci - f_X_cj).diagonal().T

    XI = Linear + Qudratic
    # 将XI中>=1的元素替换为0
    return XI < 1


def getXI_org(W, V, P, Q):
    '''
    return f(P)-f(Q)
    是一个列向量
    这个版本的XI和上一个比起来，只是返回值不一样。这个版本返回的XI是原始的f(p)-f(q)
    '''
    Z = np.dot(V, V.T)
    # p-q的线性项目
    start = time.time()
    Linear = (safe_sparse_dot((P - Q), W))

    # p-q的二次项目，暂不考虑去对角
    f_X_ci = safe_sparse_dot(safe_sparse_dot(P, Z), P.T)
    f_X_cj = safe_sparse_dot(safe_sparse_dot(Q, Z), Q.T)
    Qudratic = (f_X_ci - f_X_cj).diagonal().T

    XI = Linear + Qudratic
    # 将XI中>=1的元素替换为0
    return XI


def tah(M, XI):
    '''
    :param M 是一个梯度矩阵，每一行表示一个样本关于W的梯度向量
    :param XI 是一个列向量
    M和XI应该具有相同的行数
     1. 将XI中>=1的元素替换为0
     2. 与M进行element-wise 相乘
    '''
    assert M.shape[0] == XI.shape[0]
    return M.multiply(XI)


def update_W(old_W, batch_P, batch_Q, eta, w_lambda, input_dim, latent_dim, XI):
    cp_W = copy.deepcopy(old_W)
    # (Q-P)按行求和,合并行
    tmp = np.sum((batch_Q - batch_P).multiply(XI), axis=0)
    assert tmp.shape[0] == 1
    ## 注意，这里之前写错了cp_W-eta*(tmp.T-w_lambda*cp_W)
    cp_W = cp_W - eta * (tmp.T + w_lambda * cp_W)
    return cp_W


def update_V(old_V, batch_P, batch_Q, eta, v_lambda, input_dim, latent_dim, XI):
    '''
    :param k the row index of target element in V
    :param f the column index  of target element in V
    :param V latent matrix
    '''
    # 完全copy 一份old_V
    cp_V = copy.deepcopy(old_V)

    xi_P = batch_P.multiply(XI)
    xi_Q = batch_Q.multiply(XI)

    a = safe_sparse_dot(xi_P.T, xi_P) - safe_sparse_dot(xi_Q.T, xi_Q)
    a = safe_sparse_dot(a, cp_V)

    b = xi_P.power(2) - xi_Q.power(2)
    b = np.sum(b, axis=0)
    b = b.T
    assert b.shape == (cp_V.shape[0], 1)
    ##########
    #     print 'b:',type(b),'cp_V:',type(cp_V)
    b = np.multiply(cp_V, b)
    assert b.shape == cp_V.shape
    ##########
    deta = -(a - b)
    cp_V = cp_V - eta * (deta + v_lambda * cp_V)

    return cp_V

def init(input_dim,latent_dim):
    '''
    :return W column vector,shape:(input_dim,1)
    :return V matrix， shape:(input_dim,latent_dim)
    Note: type of both W and V is 'numpy.ndarray
    '''
#    W = np.random.uniform(low=-0.5 / input_dim, high=0.5 / input_dim, size=(input_dim,1))
    W = np.random.uniform(low=0.0 / input_dim, high=0.0 / input_dim, size=(input_dim,1))
#     W=np.random.random_sample((input_dim,1))
#     W=np.random.random_sample((input_dim,1))
    W=np.mat(W)
    V = np.random.uniform(low=-0.5/latent_dim, high=0.5/latent_dim, size=(input_dim, latent_dim))
#     V=np.random.random_sample((input_dim,latent_dim))
    V=np.mat(V)
    return W,V


def get_obj_fun_loss_matrix(W, V, X_ci, X_cj, w_lambda, v_lambda, sample_num):
    '''
     本损失是一个抽样损失，因为统计所有数据的损失比较耗时，所以每次随机抽取一定数目的样本进行统计
    '''
    total_samples = X_ci.shape[0]
    rand_index_set = np.random.randint(0, total_samples, size=sample_num)

    P = X_ci[rand_index_set]
    Q = X_cj[rand_index_set]
    #     print 'loss p.shape',P.shape
    regular = 0.5 * w_lambda * np.dot(W.T, W) + 0.5 * v_lambda * np.sum((np.multiply(V, V)))
    regular = regular[0, 0]
    loss = 0.0

    XI = getXI_org(W, V, P, Q)
    loss_vec = 1 - XI
    loss_vec = np.multiply(loss_vec, loss_vec > 0)
    loss = np.sum(loss_vec)

    return (loss + regular) / sample_num


def trian_MBGD(latent_dim, input_dim, X_ci, X_cj, w_lambda, v_lambda, eta, batch_size, epoc):
    '''
    :param latent_dim 隐向量维度
    :param input_dim 输入向量维度
    '''
    start = time.time()
    W, V = init(input_dim, latent_dim)
    print 'init is finished', '耗时', (time.time() - start), 's'

    start = time.time()
    data_size = X_ci.shape[0]

    p_start = 0
    p_end = batch_size
    total_samples = X_ci.shape[0]
    total_batches = math.ceil(total_samples / (batch_size * 1.))

    #modelPath = creatfolder()

    #     X_ci_instant,X_cj_instant=createCompleteSample()
    #     X_ci=X_ci_instant
    for ep in range(epoc):
        #             X_ci,X_cj=createCompleteSample()
        print '########Epoc:', ep, '###################'
        batch_count = 0
        p_start = 0
        p_end = batch_size
        while p_start < total_samples:
            if p_end > total_samples:
                p_end = total_samples
            # print 'p_start:',p_start,'p_end:',p_end
            batch_P = X_ci[p_start:p_end]
            batch_Q = X_cj[p_start:p_end]
            # 计算xi,这里相对与原文，没有z这一项，因为z在自己处理好的数据集上，全是1
            # 这里的XI是一个矩阵
            XI = getXI(W, V, batch_P, batch_Q)
            start = time.time()
            # 更新线性项参数
            new_W = update_W(W, batch_P, batch_Q, eta, w_lambda, input_dim, latent_dim, XI)

            # print 'new_W is finished','耗时',(time.time()-start),'s'

            start = time.time()
            # 更新二次项参数
            new_V = update_V(V, batch_P, batch_Q, eta, v_lambda, input_dim, latent_dim, XI)

            # print 'new_V is finished','耗时',(time.time()-start),'s'

            W = new_W
            V = new_V

            batch_count += 1

            p_start = p_end
            p_end = p_end + batch_size
            # 每隔1万步，输出一次loss
        #                 if (batch_count*batch_size)%10000 == 0:
        #                     sample_num=1000
        #                     obj_fun_loss=get_obj_fun_loss_matrix(W,V,X_ci,X_cj,w_lambda,v_lambda,sample_num)
        #                     print("Step:%d,loss:%.5f"%(batch_count,obj_fun_loss))
        #             print 'saving...'
        #             save_model(W,V,ep,modelPath)
        if ep!=0 and ep % 10 == 0:
            uc = predic_auc(W, np.dot(V, V.T))
            print("epoc:%d,auc:%.5f" % (ep, uc))
    return W, V

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

if __name__=="__main__":

    datapath = data_path.ml_100k
    # train_data_file = datapath + 'from_synthetic_data_csv.pkl'
    train_data_file = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/jester-2/' + 'jester-2_X_ci_X_cj.pkl'
    X_ci, X_cj = load_data_file(train_data_file)

    latent_dim=50
    input_dim=X_ci.shape[1]
#     eta=0.03
    eta = 1e-8
    w_lambda=0.1
    v_lambda=0.02
    batch_size=1000
    epoc=30
    W,V=trian_MBGD(latent_dim,input_dim,X_ci,X_cj,w_lambda,v_lambda,eta,batch_size,epoc)