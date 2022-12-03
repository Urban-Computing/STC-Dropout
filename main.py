import numpy as np
import pandas as pd
import torch
from torchsummary import summary
from models.Param import *
from models.STGCN import *
import os
import shutil
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import argparse
import utils.Metrics as Metrics
import random

random_seed=2022

random.seed(random_seed)

os.environ['PYTHONHASHSEED'] =str(random_seed)

np.random.seed(random_seed)

torch.manual_seed(random_seed)

torch.cuda.manual_seed(random_seed)

torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.deterministic =True


def getXSYS_single(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN+TIMESTEP_OUT-1:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, np.newaxis, :, :], YS[:, np.newaxis, :]
    return XS, YS

def getModel(name, args):
    ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
    A = pd.read_csv(ADJPATH).values
    W = weight_matrix(A)
    L = scaled_laplacian(W)
    Lk = cheb_poly(L, ks)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
    adj = torch.Tensor(A.astype(np.float32)).to(device)
    #adj = adj @ adj @ adj
    adj[adj>0]=1
    #print(adj.sum(1))
    model = STGCN(ks, kt, bs, T, n, Lk, p, adj, int(args.rho*N_NODE)).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def curriculum_p(p,retain=0.3, T=100, num_bz=100):
    gamma = 1000/(T*N_NODE*num_bz)
    return 1-(1-retain)*np.exp(-gamma*p)

def trainModel(name, mode, XS, YS, args):

    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)

    model = getModel(name,args)
    summary(model, (CHANNEL, TIMESTEP_IN, N_NODE), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    
    min_val_loss = np.inf
    wait = 0
    p=0
    for epoch in range(EPOCH):
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for i, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            p+=1
            pp = curriculum_p(p,retain=args.retain,T=args.T, num_bz=trainval_size/BATCHSIZE)
            if args.is_drop:
                y_pred = model(x,p=pp)
                weight = torch.zeros(x.shape[0], N_NODE).to(device)
                if i != 0:
                    idx = torch.argsort(model.score, 1, descending=True)[:, :int(pre_pp * N_NODE)]
                    idy = torch.argsort(model.score, 1, descending=True)[:, int(pre_pp * N_NODE):int(pp * N_NODE)]
                    weight = torch.scatter(weight, 1, idy, 1 + pp)
                else:
                    idx = torch.argsort(model.score, 1, descending=True)[:, :int(pp * N_NODE)]
                weight = torch.scatter(weight, 1, idx, 1).unsqueeze(1).unsqueeze(1)
                pre_pp = pp
            else:
                y_pred = model(x)
#             print(y_pred.shape)
            if args.is_drop:
                loss = criterion(y_pred * weight, y * weight)
            else:
                loss = criterion(y_pred , y )
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())

def testModel(name, mode, XS, YS , args):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, args)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)

    YS, YS_pred = np.squeeze(YS), np.squeeze(YS_pred)
    YS = scaler.inverse_transform(YS)
    YS_pred = scaler.inverse_transform(YS_pred)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Testing Ended ...', time.ctime())
        
################# Parameter Setting #######################
MODELNAME = 'STGCN'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M%S")
PATH = 'save/' + KEYWORD
###########################################################
GPU = sys.argv[-1] if len(sys.argv) == 2 else '0'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################

data = pd.read_hdf(FLOWPATH,key='data').values
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_drop', type=bool, default=True, help='whether to use STC-Dropout')
    parser.add_argument('--retain', type=float, help='the initial retain rate for STC-Dropout')
    parser.add_argument('--rho', type=float, help='the factor of determining the radius')
    parser.add_argument('--T', type=float, help='coverage epoch for vanilla model')
    args = parser.parse_args()
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
        
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS_single(data, 'TRAIN')
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS, args)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS_single(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS, args)


main()



