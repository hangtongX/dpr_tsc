import os
import  numpy as np
import torch
from torch.utils.data import  Dataset,DataLoader
import  pandas as pd
from utils.enmTypes import  InputType,EvalType,DataStatic
from functools import  partial
from joblib import  Parallel,delayed
import scipy.sparse as sp

def load_data(dataname,compression = 'gzip',):
    path = os.getcwd() + '/data/'
    train = pd.read_pickle(path + dataname + '/train.pkl',compression= compression)
    test = pd.read_pickle(path + dataname + '/test.pkl',compression= compression)
    train.columns = [ 'user','item','rating']
    test.columns = ['user', 'item', 'rating']
    info = DataStatic(dataname,int(max(train.user.max(),test.user.max())) + 1,int(max(train.item.max(),test.item.max())) + 1)
    return  train,test,info

def load_negative(dataname,compression = 'gzip'):
    path = os.getcwd() + '/data/'
    if os.access(path + dataname + '/negative.pkl',mode= os.F_OK):
        negative = pd.read_pickle(path + dataname + '/negative.pkl', compression=compression)
        return negative
    else:
        raise Exception('no exist negative file, begain to genreate.......')

class GeneralData(object):
    def __init__(self, config,validate_percent = 0.3):
        self.dataname = config.data_name
        self.train, self.test, self.datainfo = load_data(config.data_name)
        self.validate = self.split_validate(percent= validate_percent)
        self.data_type = config.data_type
        self.evalType = config.eval_type
        self.batch_size = config.batchsize
        self.num_worker = config.num_worker
        # self.padding_value = self.datainfo.item_num - 1
        self.config = config
        self.get_pop()
        if self.data_type == InputType.PAIRWISE:
            self.generatePairTrainSet()
        else:
            self.train = self.train[self.train.rating > 0]


    def  split_validate(self,percent):
        return self.test.sample(n = int(self.test.shape[0] * percent))

    def sample(self):
        try:
            data =load_negative(self.dataname)
        except Exception:
            path = os.getcwd() + '/data/'
            data = self.train.copy()
            item_pool = set(data.item.unique())
            interact_status = data.groupby('user')['item'].apply(set).reset_index().rename(columns={'item': 'interacted_items'})
            interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
            interact_status['test_negative_samples'] = interact_status['negative_items'].apply( lambda x: np.random.choice(list(x), 99,replace = False))
            data = interact_status[['user', 'interacted_items','negative_items', 'test_negative_samples']].sort_values(by = 'user', ascending= True)
            # data.to_pickle(path + self.dataname +'/negative.pkl',compression= 'gzip')
            # interact_status['negative_items'] = interact_status.apply(lambda x: (x.negative_items - set(x.test_negative_samples)), axis=1)
        return data

    def generatePairTrainSet(self):
        self.train = self.train[self.train.rating > 0]
        negatives = self.sample()
        negatives['negative'] = negatives.negative_items.apply(lambda x: np.random.choice(list(x),1,replace = False)[0])
        negatives = negatives[['user','negative']]
        self.train = pd.merge(self.train,negatives,on='user',how= 'left')
        # set columns sort
        self.train = self.train[['user','item','negative','rating']]
        if self.train.isnull().values.any():
            print('Exist NaN in Pairwise set generate process...........')
            self.train.dropna(axis= 0, how='any')

    def get_pop(self):
        item_count = self.train.groupby('item')['user'].count()
        item_count.colums = pd.Series(['count'])
        self.item_count = item_count

    def trainDataloader(self):
        dataset = GeneralDataset(self.train)
        return DataLoader(dataset= dataset, num_workers= self.num_worker, pin_memory= True,batch_size= self.batch_size,shuffle= True,prefetch_factor= 3 * self.num_worker)

    def testDataloader(self):
        test = self.test[self.test.rating > 0][['user','item']]
        shuffle = True
        if self.evalType == EvalType.FULLSORT:
            dataset = FullSortDataset(test,self.sample(),self.datainfo)
            self.max_mask_Len = dataset.max_mask_Len + 1
            self.max_pos_Len = dataset.max_pos_Len + 1
            return DataLoader(dataset=dataset, num_workers = self.num_worker, pin_memory=True, batch_size=self.batch_size, shuffle=False,collate_fn= self.collate_fn,prefetch_factor= 3 * self.num_worker)
        elif self.evalType == EvalType.NEG_SAMPLE_SORT:
            dataset = NegSampleDataset(test,self.sample())
            shuffle = False
            self.batch_size = 100
            return DataLoader(dataset=dataset, num_workers = self.num_worker, pin_memory=True, batch_size=self.batch_size, shuffle=shuffle,prefetch_factor= 3 * self.num_worker)
        else:
            raise Exception('data type is render, please check......... ')


    def validateDataloader(self):
        self.validate = self.validate[self.validate.rating > 0]
        shuffle = True
        if self.evalType == EvalType.FULLSORT:
            dataset = FullSortDataset(self.validate, self.sample(),self.datainfo)
            self.max_mask_Len = dataset.max_mask_Len + 1
            self.max_pos_Len = dataset.max_pos_Len + 1
            return DataLoader(dataset=dataset, num_workers=self.num_worker, pin_memory=True, batch_size=self.batch_size,
                              shuffle=shuffle, collate_fn=self.collate_fn)
        elif self.evalType == EvalType.NEG_SAMPLE_SORT:
            dataset = NegSampleDataset(self.validate, self.sample())
            shuffle = False
            self.batch_size = 100
            return DataLoader(dataset=dataset, num_workers=self.num_worker, pin_memory=True, batch_size=self.batch_size,
                              shuffle=shuffle)
        else:
            raise Exception('data type is render, please check......... ')


    def collate_fn(self,batch):
        user = torch.Tensor([item[0] for item in batch]).int()
        mask = torch.cat([torch.Tensor(item[1]).int().view(1,-1) for item in batch],dim= 0)
        # mask = torch.cat([torch.cat((torch.Tensor(item[1]).int().view(1,-1),torch.full((1,self.max_mask_Len - len(item[1])),self.padding_value).int()),dim= 1) for item in batch],dim= 0)
        groundtruth = torch.cat([torch.cat((torch.Tensor(item[2]).int().view(1,-1),torch.full((1,self.max_pos_Len - len(item[2])), len(item[2])).int()),dim= 1) for item in batch],dim= 0)
        return [user, mask, groundtruth]

    def createSparseGraph(self):
        path = os.getcwd() + '/data/' + self.dataname + '/sparse_graph.pt'
        if os.access(path, os.F_OK):
            return torch.load(path)
        else:
            A = sp.dok_matrix((self.datainfo.user_num + self.datainfo.item_num, self.datainfo.user_num + self.datainfo.item_num), dtype=np.float32)
            UserItemNet = sp.coo_matrix((np.ones(len(self.train['user'])), (self.train['user'], self.train['item'])),
                                     shape=(self.datainfo.user_num, self.datainfo.item_num))
            UserItemNet_t = UserItemNet.transpose()
            data_dict = dict(
                zip(zip(self.train.user, self.train.item + self.datainfo.user_num), [1] * self.train.shape[0])
            )
            data_dict.update(
                dict(
                    zip(
                        zip(self.train.user + self.datainfo.user_num, self.train.item),
                        [1] * self.train.shape[1],
                    )
                )
            )
            A._update(data_dict)
            # norm adj matrix
            sumArr = (A > 0).sum(axis=1)
            # add epsilon to avoid divide by zero Warning
            diag = np.array(sumArr.flatten())[0] + 1e-7
            diag = np.power(diag, -0.5)
            D = sp.diags(diag)
            L = D * A * D
            # covert norm_adj matrix to tensor
            L = sp.coo_matrix(L)
            row = L.row
            col = L.col
            i = torch.LongTensor(np.array([row, col]))
            data = torch.FloatTensor(L.data)
            SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
            torch.save(SparseL,path)
            return SparseL



    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))



class GeneralDataset(Dataset):
    def __init__(self,data):
        self.data = data.to_numpy()

    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        return self.data[id][0].astype(np.int32),self.data[id][1].astype(np.int32),self.data[id][2].astype(np.int32)




class FullSortDataset(Dataset):
    def __init__(self,data,interaction,info):
        self.info = info
        self.data,self.groundTruth,self.mask,self.max_mask_Len, self.max_pos_Len = self.generate(data,interaction)

    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        return self.data[id],list(self.mask[self.data[id]]),list(self.groundTruth[self.data[id]])

    def generate(self,data,interaction):
        mask = interaction[['user','interacted_items']].set_index('user').to_dict(orient = 'dict')['interacted_items']
        max_maxk_Len = interaction.interacted_items.apply(len).max()
        groundTruth = data.groupby('user')['item'].apply(set).reset_index().rename(columns={'item': 'groundTruth'})[['user','groundTruth']]\
            .set_index('user').to_dict(orient = 'dict')['groundTruth']
        max_pos_Len = max([len(j) for j in groundTruth.values()])
        for user in (set(groundTruth.keys()) - set(mask.keys())):
            mask[user] = set([])
        for user in mask.keys():
            item = np.array(list(mask[user])).astype('int')
            a = np.ones(self.info.item_num)
            a[item] = 0
            mask[user] = a.tolist()
        test = data.user.unique()

        return test,groundTruth,mask,max_maxk_Len,max_pos_Len





class NegSampleDataset(Dataset):
    def __init__(self,data,interaction):
        self.data = self.generate(data,interaction)

    def  __len__(self):
        return self.data.shape[0]

    def __getitem__(self, id):
        return self.data[id][0],self.data[id][1]

    def generate(self,data,interaction):
        process = partial(self.applyfunc,negatives = interaction[['user','test_negative_samples']].set_index('user').to_dict(orient = 'dict')['test_negative_samples'])
        return self.applyParallel(data,process)

    def applyfunc(self,x,negatives):
        a = np.array(list(negatives[x[0]])).reshape(-1,1)
        b = np.full_like(a,x[0]).reshape(-1,1)
        return np.vstack((x,np.hstack((b,a))))

    def applyParallel(self,data,func):
        res = Parallel(n_jobs=-1, verbose=0, backend='threading', timeout=10)(delayed(func)(value) for value in data.values)
        return np.concatenate(res)




