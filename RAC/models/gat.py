from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)  # Single head in final layer
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First GAT layer with multi-head attention
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        return x  # Raw logits for multi-class classification
