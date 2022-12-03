import pickle
import torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
def masked_mae(preds, labels, null_val=0.0):
    preds[preds<1e-5]=0
    labels[labels<1e-5]=0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

class Outlier_detection(nn.Module):
    def __init__(self, k=62):
        super(Outlier_detection, self).__init__()
        self.k=k
    def get_distance(self, data):
        '''
        data [bz, channel, H, W]
        X [bz, H*W, channel]

        return [bz, H*W, H*W]
        '''
        X = data.detach().reshape(len(data), data.shape[1]*data.shape[2], -1).permute(0, 2, 1)
        X_std = torch.std(X, dim=1, keepdim=True)
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X = (X - X_mean) / (X_std + 1e-6)
        distance = torch.cdist(X, X, p=2)
        return distance
    def forward(self, data, p):
        ##lof
#         distance = self.get_distance(data).cpu().detach().numpy()
#         for i, sub_distance in enumerate(distance):
#             if 1-p==0: break
#             clf = LocalOutlierFactor(n_neighbors=self.k, contamination=1-p,metric='precomputed')
#             y_pred = clf.fit_predict(sub_distance)
#             X_scores = clf.negative_outlier_factor_
#             data[i,:,:,X_scores==-1]=0
        ##if
        X = data.detach().reshape(len(data), data.shape[1]*data.shape[2], -1).permute(0, 2, 1)
        for i, x in enumerate(X):
            if 1-p==0: break
            clf = IsolationForest(n_estimators=10, warm_start=True, contamination=1-p)
            y_pred=clf.fit_predict(x.cpu().detach().numpy())
            data[i,:,:,y_pred==-1]=0
        return data

class STDrop(nn.Module):

    def __init__(self, adj, k=30):
        super(STDrop, self).__init__()
        # self.adj = self.get_neighbor()
        self.adj = adj
        self.k = k

    def get_neighbor(self, h=10, w=20):
        adj = torch.zeros((h * w, h * w)).cuda()
        for i in range(h):
            for j in range(w):
                for u in range(h):
                    for v in range(w):
                        adj[i * w + j, u * w + v] = (i - u) ** 2 + (j - v) ** 2
        adj[adj > 3] = 0
        adj[adj != 0] = 1
        return adj

    def get_distance(self, data):
        '''
        data [bz, channel, H, W]
        X [bz, H*W, channel]

        return [bz, H*W, H*W]
        '''
        X = data.detach().reshape(len(data), data.shape[1]*data.shape[2], -1).permute(0, 2, 1)
        X_std = torch.std(X, dim=1, keepdim=True)
        X_mean = torch.mean(X, dim=1, keepdim=True)
        X = (X - X_mean) / (X_std + 1e-6)
        distance = torch.cdist(X, X, p=2)
        return distance

    def get_score(self, distance):
        '''
        distance [bz, H*W, H*W]
        '''
        sort_distance = torch.sort(distance, axis=2)[0]
        # sort_distance [bz, H*W, H*W]
        batch_R = sort_distance.mean(axis=1)[:, self.k]
        # [bz]
        adj_distance = distance * self.adj
        # [bz. H*W, H*W]
        adj_distance[adj_distance == 0] = 1e10

        temp_distance = distance.clone()
        temp_adj_distance = adj_distance.clone()
        self.samples_N = torch.zeros_like(sort_distance.mean(axis=1)).to(sort_distance.device)
        self.neighbor_N = torch.zeros_like(sort_distance.mean(axis=1)).to(sort_distance.device)
        # [bz,H*W]
        for i, x in enumerate(temp_adj_distance):
            x[x < batch_R[i]] = -1
            x[x > 0] = 0
            x[x == -1] = 1
            self.neighbor_N[i] = x.sum(axis=1)
        for i, x in enumerate(temp_distance):
            x[x < batch_R[i]] = -1
            x[x > 0] = 0
            x[x == -1] = 1
            self.samples_N[i] = x.sum(axis=1)
        spatial_score = self.neighbor_N/self.adj.sum(-1)
        temporal_score =  self.samples_N / (self.samples_N + self.samples_N.mean(axis=-1,keepdim=True))
        score = 2 - spatial_score - temporal_score

        return score

    def forward(self, data, p=0.5):
        distance = self.get_distance(data)
        score = self.get_score(distance)
        total_score = score.clone()
        score = score.argsort(dim=1,descending=False).argsort(dim=1,descending=False)
        score[score<score.shape[1]*p] = -1
        score[score>-1]= 0
        #score = score/p
        score = score.view(data.shape[0], 1, 1, data.shape[3])#.repeat(1, data.shape[1], 1, 1)
        data = data * score * -1
        return data, total_score