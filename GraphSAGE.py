from dgl.nn.pytorch.conv import SAGEConv

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    mask = torch.BoolTensor(data.train_mask)
    g = DGLGraph(data.graph)
    n_classes = data.num_classes
    return n_classes, g, features, labels, mask

n_classes, g, features, labels, mask = load_cora_data()

# create GraphSAGE model
model = GraphSAGE(in_feats=features.size()[1],
                  n_hidden=16,
                  n_classes=n_classes,
                  n_layers=1,
                  activation=F.relu,
                  dropout=0.5,
                  aggregator_type='gcn')

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# initialize graph
dur = []
for epoch in range(50):
    model.train()
    if epoch >= 3:
        t0 = time.time()

    logits = model(g, features)
    loss = F.cross_entropy(logits[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item()), np.mean(dur))
