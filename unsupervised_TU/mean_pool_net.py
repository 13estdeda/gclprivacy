import torch
import torch.nn.functional as F
from torch.nn import Linear

from diffpool_net import GNN
import numpy as np


class MeanPoolNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, hidden_channels=64):
        super(MeanPoolNet, self).__init__()
        self.conv = GNN(num_feats, hidden_channels, hidden_channels, lin=False)
        
        self.lin1 = Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv(x, adj, mask))
        
        self.graph_embedding = x.mean(dim=1)
        
        x = F.relu(self.lin1(self.graph_embedding))
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)


class MeanPoolNet_encoder(torch.nn.Module):
    def __init__(self, num_feats, num_classes, hidden_channels=32):
        super(MeanPoolNet_encoder, self).__init__()
        self.conv = GNN(num_feats, hidden_channels, hidden_channels, lin=False)
        self.proj_head = torch.nn.Sequential(torch.nn.Linear(3 * hidden_channels, 3 * hidden_channels), torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(3 * hidden_channels, 3 * hidden_channels))

        self.lin1 = Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x, adj, mask=None, is_head=False):
        x = F.relu(self.conv(x, adj, mask))

        self.graph_embedding = x.mean(dim=1)

        # x = F.relu(self.lin1(self.graph_embedding))
        # x = self.lin2(x)
        if is_head:
            self.graph_embedding=self.proj_head(self.graph_embedding)

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