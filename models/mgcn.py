import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class ResidualMLP(nn.Module):
    """
    4-layer MLP with Residual Blocks.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super(ResidualMLP, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 2 (Residual)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 3 (Residual)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Layer 4 (Output)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Block 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Block 2 (Residual)
        identity = x
        out = self.fc2(x)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        x = out + identity # Residual connection
        
        # Block 3 (Residual)
        identity = x
        out = self.fc3(x)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.dropout(out)
        x = out + identity # Residual connection
        
        # Block 4
        out = self.fc4(x)
        return out

class MGCN_CL(nn.Module):
     
    def __init__(self, input_dim, hidden_dim, num_classes, pooling_ratio=0.5):
        super(MGCN_CL, self).__init__()
        
        self.pooling_ratio = pooling_ratio
        
        # --- Parallel Multi-scale GCN Module ---
        
        # Branch 1: Scale-1 (Local Texture) - 1 GCN layer
        self.scale1_gcn = GCNConv(input_dim, hidden_dim)
        
        # Branch 2: Scale-2 (Global Semantics) - 2 cascaded GCN layers
        self.scale2_gcn1 = GCNConv(input_dim, hidden_dim)
        self.scale2_gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Branch 3: Raw Feature (Petrophysical Anchor) - Linear Projection
        self.raw_proj = nn.Linear(input_dim, hidden_dim)
        
        # Fusion dimensionality: 3 * hidden_dim
        self.fusion_dim = 3 * hidden_dim
        
         
        self.gcn_pos = GCNConv(self.fusion_dim, 1)
         
        self.gcn_neg = GCNConv(self.fusion_dim, 1)
        
        
        self.pool_transform = nn.Linear(self.fusion_dim, self.fusion_dim)

         
        self.readout_dim = 2 * self.fusion_dim
        
        
        self.classifier = ResidualMLP(self.readout_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (single graph), create batch vector of zeros
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

         
        h_s1 = self.scale1_gcn(x, edge_index)
        h_s1 = F.leaky_relu(h_s1)
        
         
        h_s2_step1 = self.scale2_gcn1(x, edge_index)
        h_s2_step1 = F.leaky_relu(h_s2_step1)
        h_s2 = self.scale2_gcn2(h_s2_step1, edge_index)
        h_s2 = F.leaky_relu(h_s2)
        
         
        h_raw = self.raw_proj(x)
        
        h_final = torch.cat([h_s1, h_s2, h_raw], dim=1)
        
        
        score_neg = torch.sigmoid(self.gcn_neg(h_final, edge_index)).view(-1)
        # S+ = sigmoid(GCN+(H_final))
        score_pos = torch.sigmoid(self.gcn_pos(h_final, edge_index)).view(-1)
        
        
        (batch_pos, x_pos, 
         mask_pos) = self._apply_pooling(h_final, score_pos, batch, self.pooling_ratio)
        
         
        (batch_neg, x_neg, 
         mask_neg) = self._apply_pooling(h_final, score_neg, batch, self.pooling_ratio)
        
         
        embed_pos = self._global_readout(x_pos, batch_pos)
         
        logits_pos = self.classifier(embed_pos)
        probs_pos = F.softmax(logits_pos, dim=1)
        
        
        embed_neg = self._global_readout(x_neg, batch_neg)
        logits_neg = self.classifier(embed_neg)
        probs_neg = F.softmax(logits_neg, dim=1)
        
        return logits_pos, probs_pos, probs_neg, score_pos, score_neg, batch

    def _apply_pooling(self, x, scores, batch, ratio):
       
        
        num_nodes_per_graph = torch.bincount(batch)
         
        k_per_graph = (num_nodes_per_graph.float() * ratio).ceil().long()
        
        
        mask = torch.zeros_like(scores, dtype=torch.bool)
        
         
        unique_batches = torch.unique(batch)
        for b_id in unique_batches:
            node_indices = (batch == b_id).nonzero(as_tuple=True)[0]
            k = k_per_graph[b_id]
            if k == 0: k = 1 # Ensure at least one node
            
            graph_scores = scores[node_indices]
            _, topk_local_idx = torch.topk(graph_scores, k)
            topk_global_idx = node_indices[topk_local_idx]
            
            mask[topk_global_idx] = True
            
        # Select features
        x_pooled = x[mask]
        batch_pooled = batch[mask]
        
         
        x_transformed = torch.sigmoid(self.pool_transform(x_pooled))
        
        return batch_pooled, x_transformed, mask

    def _global_readout(self, x, batch):
         
        mean_pool = global_mean_pool(x, batch)
        
        max_pool = global_max_pool(x, batch)
        
        out = torch.cat([mean_pool, max_pool], dim=1)
        return out
