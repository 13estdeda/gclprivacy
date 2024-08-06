from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
# from torch_geometric.nn import dense_mincut_pool
from dense_mincut_pool import dense_mincut_pool

from diffpool_net import GNN
import numpy as np

class MinCutPoolNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, max_nodes, hidden_channels=64):
        super(MinCutPoolNet, self).__init__()

        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        self.conv1 = GNN(num_feats, hidden_channels, hidden_channels, lin=False)
        num_nodes = ceil(0.5 * max_nodes)
        self.pool1 = Linear(3 * hidden_channels, num_nodes)

        # self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(3 * hidden_channels, num_nodes)

        # self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv3 = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.lin1 = Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    # def forward(self, x, edge_index, batch):
    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv1(x, adj))

        # x, mask = to_dense_batch(x, batch)
        # adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)
        
        # if torch.any(x.isnan()) or torch.any(adj.isnan()):
        #     a = 1

        x = F.relu(self.conv2(x, adj))
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        self.graph_embedding = x.mean(dim=1)
        
        x = F.relu(self.lin1(self.graph_embedding))
        x = self.lin2(x)
        
        # return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2
        return F.log_softmax(x, dim=-1)


class MinCutPoolNet_encoder(torch.nn.Module):
    def __init__(self, num_feats, num_classes, max_nodes, hidden_channels=32):
        super(MinCutPoolNet_encoder, self).__init__()

        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        self.conv1 = GNN(num_feats, hidden_channels, hidden_channels, lin=False)
        num_nodes = ceil(0.5 * max_nodes)
        self.pool1 = Linear(3 * hidden_channels, num_nodes)

        # self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(3 * hidden_channels, num_nodes)

        # self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv3 = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)
        self.proj_head = torch.nn.Sequential(torch.nn.Linear(3 * hidden_channels, 3 * hidden_channels),
                                             torch.nn.ReLU(inplace=True),
                                             torch.nn.Linear(3 * hidden_channels, 3 * hidden_channels))

        self.lin1 = Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    # def forward(self, x, edge_index, batch):
    def forward(self, x, adj, mask=None,is_head=False):
        x = F.relu(self.conv1(x, adj))

        # x, mask = to_dense_batch(x, batch)
        # adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        # if torch.any(x.isnan()) or torch.any(adj.isnan()):
        #     a = 1

        x = F.relu(self.conv2(x, adj))
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        self.graph_embedding = x.mean(dim=1)
        if is_head:
            self.graph_embedding=self.proj_head(self.graph_embedding)

        # x = F.relu(self.lin1(self.graph_embedding))
        # x = self.lin2(x)

        # return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2
        return self.graph_embedding

    def get_embedding(self, loader,device,DS):
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, adj, mask = data.x, data.adj, data.mask
                if DS == 'MUTAG' or DS == 'AIDS':
                    adj=adj[:,:,:,0]


                if x is None:
                    x = torch.ones((data.batch.shape[0],1)).to(device)

                x = self.forward(x, adj, mask)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
            ret = np.concatenate(ret, 0)
            y = np.concatenate(y, 0)
        return ret, y
