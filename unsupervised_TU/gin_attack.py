import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *
from sklearn import preprocessing
# from torch_geometric.datasets import TUDataset
from aug import TUDataset_aug as TUDataset
from tu_dataset_ori import  TUDataset as TUDataset_ori
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim
from sklearn.metrics import accuracy_score
from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from model import propty_attack

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from diffpool_net import DiffPoolNet
from gin import Net
def convert_to_one_hot(Y,C):
    Y=np.eye(C)[Y.reshape(-1)]
    return Y
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MyFilter(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def __call__(self, data):
        return data.num_nodes <= self.max_nodes





import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def tensor_edge_to_numpy_adj(edge_index):
    node_num=torch.max(edge_index).item()
    # print(node_num)
    adj=np.zeros((node_num+1,node_num+1))
    for i in range(edge_index.size()[1]):
        adj[edge_index[0][i]][edge_index[1][i]]=1
    return adj


def cal_density(adj):
    num_node=adj.shape[0]
    num_edge=np.count_nonzero(adj)/2
    return num_edge/(num_node*num_node)

def cal_node(adj):
    num_node=adj.shape[0]

    return num_node

def _generate_bin(attr, num_class):
    sort_attr = np.sort(attr)
    bins = np.zeros(num_class - 1)
    unit = attr.size / num_class
    for i in range(num_class - 1):
        bins[i] = (sort_attr[int(np.floor(unit * (i + 1)))] + sort_attr[int(np.ceil(unit * (i + 1)))]) / 2

    return bins




if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(42)

    accuracies = {'val':[], 'test':[]}
    epochs = 20
    log_interval = 10
    batch_size = 1
    # num_class=2
    # batch_size = 512
    # lr = args.lr
    # DS = 'MUTAG'
    # DS = 'PROTEINS'
    DS = 'AIDS'
    # DS = 'DD'
    # DS = 'NCI1'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    dataset_eval = TUDataset(path, name=DS, aug='none')
    num_class = dataset_eval.num_classes

    try:
        dataset_num_features = dataset_eval.get_num_feature()
    except:
        dataset_num_features = 1


    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size,shuffle=False)


    test_propty = []

    max_nodes=0

    for data in dataloader_eval:
        data,_=data
        edge_index=data.edge_index
        adj=tensor_edge_to_numpy_adj(edge_index)
        density=cal_node(adj)
        if density>=max_nodes:
            max_nodes=density
        test_propty.append(density)


    test_propty = np.array(test_propty)

    # print(train_propty)

    bins = _generate_bin(test_propty, num_class)
    # train_label = np.digitize(train_propty, bins)
    test_label = np.digitize(test_propty, bins)
    # print(train_label)



    # sys.exit()

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    dataset_gin = TUDataset_ori(path, name=DS)
    train_loader = DataLoader(dataset_gin, batch_size=128)
    model = Net(dataset_gin.data.num_features,2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        loss_all=0
        correct=0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

            pred=output.max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()



        print('epoch:',epoch,'loss:',loss_all,'acc:',correct/len(test_propty))



    # sys.exit()
    output_all=[]
    for data in train_loader:
        data = data.to(device)

        output = model(data.x, data.edge_index, data.batch)
        output_all.extend(output.detach().cpu().numpy())
    output_all = np.array(output_all)


    emb_train=torch.from_numpy(output_all[:int(len(output_all)/2)])
    emb_train=emb_train.type(torch.FloatTensor)
    emb_test=torch.from_numpy(output_all[int(len(output_all)/2):])
    emb_test=emb_test.type(torch.FloatTensor)

    train_label=torch.from_numpy(test_label[:int(len(output_all)/2)])
    train_label=train_label.type(torch.LongTensor)
    test_label=torch.from_numpy(test_label[int(len(output_all)/2):])
    test_label=test_label.type(torch.LongTensor)

    train_data_label = torch.utils.data.TensorDataset(emb_train, train_label)
    test_data_label = torch.utils.data.TensorDataset(emb_test, test_label)




    if DS=='DD':
        train_loader = torch.utils.data.DataLoader(dataset=train_data_label, batch_size=64,
                                                   shuffle=True, num_workers=1, drop_last=False)
        validate_loader = torch.utils.data.DataLoader(dataset=test_data_label, batch_size=64, shuffle=False,
                                                      num_workers=1, drop_last=False)
    else :
        train_loader = torch.utils.data.DataLoader(dataset=train_data_label, batch_size=128,
                                                   shuffle=True, num_workers=1, drop_last=False)
        validate_loader = torch.utils.data.DataLoader(dataset=test_data_label, batch_size=128, shuffle=False,
                                                      num_workers=1, drop_last=False)



    # print(emb_train.shape)
    # print(train_label.shape)
    #
    # train_data=[]








    attack_model=propty_attack(output_all.shape[1],num_class)
    opt_attack=torch.optim.Adam(attack_model.parameters(), lr=0.01)

    best_acc = 0
    std=[]
    for epoch in range(epochs):
        loss_all=0.0
        predict_label_all = []
        ori_label_all = []
        for index,(data,labels) in enumerate(train_loader):

            attack_model.train()
            opt_attack.zero_grad()
            predict=attack_model(data)
            predict_att=torch.max(predict,1)[1]

            loss = F.cross_entropy(predict, labels)
            loss_all+=loss.item()
            loss.backward()

            opt_attack.step()
            attack_model.eval()

            predict_att = predict_att.cpu().numpy()

            labels = labels.cpu().numpy()
            predict_label_all.extend(predict_att)
            ori_label_all.extend(labels)
        predict_label_all = np.array(predict_label_all)
        ori_label_all = np.array(ori_label_all)

        acc = 0
        for i in range(len(predict_label_all)):
            if predict_label_all[i] == ori_label_all[i]:
                acc += 1

        print('train_acc:', acc / len(predict_label_all))




        print('epoch:', epoch, 'loss_dis:', loss_all / (index + 1))



        predict_label_all = []
        ori_label_all = []
        for index, (data, labels) in enumerate(validate_loader):
            attack_model.eval()
            predict_logit = attack_model(data)
            predict_label = torch.max(predict_logit, 1)[1]
            predict_label = predict_label.cpu().numpy()

            labels = labels.cpu().numpy()
            predict_label_all.extend(predict_label)
            ori_label_all.extend(labels)
        predict_label_all = np.array(predict_label_all)
        ori_label_all = np.array(ori_label_all)
        acc = 0
        for i in range(len(predict_label_all)):
            if predict_label_all[i] == ori_label_all[i]:
                acc += 1
        test_acc=acc / len(predict_label_all)
        print('test_acc:', acc / len(predict_label_all))

        if test_acc>=best_acc:
            best_acc=test_acc
        if epoch>=(epochs-20):
            std.append(test_acc)







    std=np.array(std)
    std=np.std(std)
    print('best_acc:',best_acc)
    print('std:',std)


    # for data in dataloader:
    #     data, data_aug = data





    # tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    #
    # if not os.path.exists('logs/log_' + args.DS + '_' + args.aug):
    #     os.makedirs('logs/log_' + args.DS + '_' + args.aug)
    # with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
    #     s = json.dumps(accuracies)
    #     f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
    #     f.write('\n')
