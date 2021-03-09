import dgl
import numpy as np

# discribe the graph with DGL
def build_karate_club_graph():
    # 所有 78 条边都存储在两个 numpy 数组中 , 一个用于源端点而另一个用于目标端点
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10,
    10,
    10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
    25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
    33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
    31, 32])
    # 边缘在 DGL 中是有方向的; 使它们双向
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # 构建图
    return dgl.DGLGraph((u, v))

G = build_karate_club_graph()
print('We have %d nodes.' % G . number_of_edges())
print('We have %d edges.' % G . number_of_nodes())

import networkx as nx
import matplotlib.pyplot as plt
# Since the actual graph is undirected, we convert it for visualization
# purpose.
nx_G = G.to_networkx().to_undirected()
# Kamada‐Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
plt.savefig('graph.png')
plt.show()

# assign features to nodes and edges
import torch
import torch.nn as nn
import torch.nn.functional as F

embed = nn.Embedding(34, 5)  # 34 nodes with embedding dim equal to 5
G.ndata['feat'] = embed.weight

# create GCN model
from dgl.nn.pytorch import GraphConv

class GCN (nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h
        
# 第一层将大小为 5 的输入特征转换为大小为 5 的隐藏层
# 第二层将因隐藏层转换为大小为 2 的输入特征,相当于两组俱乐部
net = GCN(5, 5, 2)
print('net: ', net)

# initial data
inputs = embed.weight
labeled_nodes = torch.tensor([0,33])
labels = torch.tensor([0,1])

# train and visualization
import itertools

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
for epoch in range(50):
    logits = net(G, inputs)
    # 保存 logits 主要为了后续进行可视化实现
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # 只计算标签节点的损失
    loss = F.nll_loss(logp[labeled_nodes], labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
