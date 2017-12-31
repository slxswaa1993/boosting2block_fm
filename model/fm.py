# coding:utf8
import numpy as np
import random
import scipy.sparse as sp
from sklearn.utils.extmath import safe_sparse_dot
import pickle
import time

def predict(x,w0,W,V,sampler):
    '''
    :param x: (d,1) NOTE: the shape should not be (d,)
    :param w0:
    :param W:
    :param V:
    :return:
    如果x里面只有两个维度不是0，那么二次项计算，实际上只是这两个维度的交互。
    1. 获取x中非零元素下标
    2. 获取
    '''
    #ndarray [a,b]
    indx_feature=np.where(x==1)[0]
    num_features=len(indx_feature)

    qudratic_term=0.0
    for i in xrange(0,num_features-1):
        for j in xrange(i+1,num_features):
            v_feacture_1 = V[indx_feature[i]]
            v_feacture_2 = V[indx_feature[j]]
            qudratic_term += np.dot(v_feacture_1,v_feacture_2.T)

    linear_term = np.sum(W[indx_feature])

    y= w0 + linear_term + qudratic_term

    ## 回归的时候，确保不超出范围
    y = max(sampler.min_target,y)
    y = min(sampler.max_target, y)

    return y


def predict_fast(test_data,w0,W,V,sampler):
    '''
    :param test_data: sparse csr_matrix shape:(N,totaldimension)
    :param w0:
    :param W:
    :param V:
    :return:
    如果x里面只有两个维度不是0，那么二次项计算，实际上只是这两个维度的交互。
    1. 获取x中非零元素下标
    2. 获取
    '''
    #ndarray [a,b]

    Z = np.dot(V, V.T)
    temp=safe_sparse_dot(safe_sparse_dot(test_data, Z), test_data.T).diagonal().T
    # NOTE test_data.T should be test_data.T**2,but all the value is eigther 0 or 1.
    temp2=safe_sparse_dot(Z.diagonal(),test_data.T)
    qudratic_term=0.5*(temp-temp2)
    y = w0 + safe_sparse_dot(test_data, W) + qudratic_term
    ## 回归的时候，确保不超出范围
    y = max(sampler.min_target,y)
    y = min(sampler.max_target, y)

    return y


def update_factors(w0,W,V,reg_w,reg_v,eta,x,y,sampler,wathcer):
    '''
    目前只实现regression,采用 least square loss
    :param w0:the bias
    :param W:ndarray (D,1),where D is the dimensions of input vector
    :param V: ndarray (D,f), where f is the number of factors of latent vector
    :param reg_w:
    :param reg_v:
    :param eta:
    :param x : (d,1) NOTE: the shape should not be (d,)
    :param y : real labels
    :return:
    '''
    assert x.shape[1]==1

    y_p=predict(x,w0,W,V,sampler)

    error=2*(y_p-y)
    wathcer.addError(np.abs(error))
    # update bais:
    w0-=eta*(1*error+2*w0*reg_w)

    indx_feature = np.where(x == 1)[0]
    num_features=len(indx_feature)
    num_factors=V.shape[1]

    # update w
    W[indx_feature]-=eta*(x[indx_feature]*error+2*W[indx_feature]*reg_w)


    # update V
    # V的更新实际上也涉及到一些矩阵相乘，但是其实，每次只有不为零的那些维度对应的V需要更新
    v_deat=np.zeros((num_features,num_factors))
    for col in xrange(0,num_factors):
        v_deat[:,col] = np.sum(V[:, col][indx_feature])
        v_deat[:,col] -= V[:,col][indx_feature]

    V[indx_feature] -= eta*(v_deat*error+2*V[indx_feature]*reg_v)


def init(num_factors,num_attribute,init_stdev,seed):
    W = np.zeros(num_attribute).reshape(-1,1)
    np.random.seed(seed=seed)
    V = np.random.normal(scale=init_stdev, size=(num_attribute, num_factors))
    return W,V

# todo
def loss():
    return 0.0

class Watcher(object):
    def __init__(self):
        self.train_loss=[]

    def addError(self,s_error):
        self.train_loss.append(s_error)

    def mean_loss(self):
        return np.average(self.train_loss)

    def clear_loss(self):
        self.train_loss = []
 #
 # X_train, validation, train_labels, validation_labels = cross_validation.train_test_split(
 #            X, y, test_size=self.validation_size)

def train(sampler,hyper_args):
    """train model
    data: user-item matrix as a scipy sparse matrix
          users and items are zero-indexed
    """
    global W, V,w0

    reg_w=hyper_args['reg_w']
    reg_v=hyper_args['reg_v']
    eta=hyper_args['eta']
    num_factors=hyper_args['num_factors']
    w0=hyper_args['w0']
    num_iters=hyper_args['num_iters']
    init_stdev = hyper_args['init_stdev']
    seed=hyper_args['seed']

    num_users=sampler.num_users
    num_items=sampler.num_items

    # 这里将遗漏两个item
    total_dim=num_users+num_items

    W, V = init(num_factors,total_dim,init_stdev,seed)

    wathcer=Watcher()
    print 'initial loss = {0}'.format(loss())
    for it in xrange(num_iters):
        print 'starting iteration {0}'.format(it)
        sample_count=0
        for u,i,y in sampler.generate_samples():

            if sampler.num_users+i >= 2623:
                print "IndexError i={0}".format(i)
                continue

            # NOTE,userID and itemID should be 0-index
            x=np.zeros((total_dim,))
            x[u]=1
            x[sampler.num_users+i]=1
            x=x.reshape(-1,1)
            update_factors(w0,W,V,reg_w,reg_v,eta,x,y,sampler,wathcer)
            sample_count+=1
            if sample_count % 1e3 ==0:
                print "Handle {0} of {1} samples".format(sample_count,sampler.num_samples)
                print "loss={0}".format(wathcer.mean_loss())
                wathcer.clear_loss()

        print 'iteration {0}: loss = {1}'.format(it,loss())
    return (w0,W,V)


def loadData(filename, path="/Users/dong/Desktop/BoostingFM-IJCAI18/dataset/ml-100k/",is_0_index=False):
    data = []
    y = []
    users = set()
    items = set()
    with open(path + filename) as f:
        for line in f:
            (user, movieid, rating, ts) = line.split('\t')
            user_id = int(user)
            movie_id = int(movieid)
            if not is_0_index:
                user_id -= 1
                movie_id -= -1

            data.append({"user_id": user_id, "movie_id": movie_id})
            y.append(float(rating))
            users.add(user_id)
            items.add(movie_id)

    return (np.array(data), np.array(y), users, items)


class Data_Generator(object):

    def __init__(self,train_data,train_y,users,items,max_samples=None,isShuffle=True):
        data_size = len(train_y)
        if max_samples is None:
            self.num_samples = data_size
        else:
            self.num_samples = min(data_size, max_samples)

        self.train_data=train_data
        self.train_y=train_y
        self.num_users=len(users)
        self.num_items=len(items)
        # 回归的时候，预测的值不能超出范围
        self.min_target = min(train_y)
        self.max_target = max(train_y)
        # 打乱数据
        if isShuffle:
            idxs = range(self.num_samples)
            random.shuffle(idxs)
            self.train_data = train_data[idxs]
            self.train_y = train_y[idxs]

    def generate_samples(self):
        idx=0
        for _ in xrange(self.num_samples):
            u = self.train_data[idx]['user_id']
            i = self.train_data[idx]['movie_id']
            y = self.train_y [idx]
            idx += 1
            yield u, i, y


def predict_all(sampler):
    num_users = sampler.num_users
    num_items = sampler.num_items

    # 这里将遗漏两个item
    total_dim = num_users + num_items
    watcher=Watcher()
    for u, i, y in sampler.generate_samples():
        if sampler.num_users + i >= 2623:
            print "IndexError i={0}".format(i)
            continue
        # NOTE,userID and itemID should be 0-index
        x = np.zeros((total_dim,))
        x[u] = 1
        x[sampler.num_users + i] = 1
        x = x.reshape(-1, 1)
        y_p = predict(x, w0, W, V, sampler)
        watcher.addError(np.abs(y-y_p))

    return watcher.mean_loss(),watcher.train_loss

if __name__=="__main__":

    train_file="ua.base"
    train_data,train_y,users,items= loadData(train_file)

    sampler = Data_Generator(train_data,train_y,users,items)

    hyper_args={
    "reg_w":0.0025,
    "reg_v":0.0025,
    "eta":0.01,
    "num_factors":50,
    "w0":0.0,
    "num_iters":1,
    "init_stdev":0.1,
    "seed":28
    }

    w0, W, V=train(sampler, hyper_args)

    test_file = "ua.test"
    test_data, test_y, _ , _ = loadData(test_file)

    sampler_pred = Data_Generator(test_data, test_y, users, items)

    preds, preds= predict_all(sampler_pred)
    from sklearn.metrics import mean_squared_error
    print("FM RMSE: %.4f" % np.sqrt(mean_squared_error(test_y, preds)))

    with open("weight.pkl","wb") as fi:
        pickle.dump(w0,fi)
        pickle.dump(W,fi)
        pickle.dump(V,fi)

