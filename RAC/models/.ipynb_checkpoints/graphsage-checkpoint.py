import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 128)
        self.mlp = nn.Linear(128, out_channels) 


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embedding = self.conv2(x, edge_index)
        x = self.mlp(embedding)
        return x , F.normalize(embedding, p=2, dim=-1)# Raw logits for multi-class classification F.log_softmax(x, dim=1)
