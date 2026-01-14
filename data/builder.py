import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import scipy.signal as signal
import numpy as np

class GraphBuilder:
   
    def __init__(self, p_blocks=8, kappa=0.6, wavelet='mexh', scales=None):
       
        self.p_blocks = p_blocks
        self.kappa = kappa
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(1, 129)

    def compute_cwt(self, signal_data):
        """
        Args:
            signal_data: (Length, ) or (Length, Channels)
        Returns:
            cwt_matrices: List of CWT matrices [ (Freq, Time), ... ] per channel
        """
        # Ensure signal_data is 2D: (Length, Channels)
        if signal_data.ndim == 1:
            signal_data = signal_data[:, np.newaxis]
            
        num_channels = signal_data.shape[1]
        cwt_list = []
        
        for c in range(num_channels):
            channel_signal = signal_data[:, c]
            # scipy.signal.cwt returns (len(scales), len(data))
            coefficients = signal.cwt(channel_signal, signal.ricker, self.scales)
            cwt_list.append(np.abs(coefficients))
            
        return cwt_list 

    def normalize(self, matrix):
        """
        Min-Max scale to [-1, 1]
        """
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val == 0:
            return np.zeros_like(matrix)
         
        scaled = (matrix - min_val) / (max_val - min_val)
        return 2 * scaled - 1

    def build_graph(self, raw_signal, label=None):
        """
        Args:
            raw_signal: (Window_Size, Num_Channels)
        """
        
        # 1. Compute CWT for all channels
        cwt_matrices = self.compute_cwt(raw_signal)
        
        # 2. Normalize each channel's CWT map
        cwt_matrices = [self.normalize(m) for m in cwt_matrices]
        
        # Shape of one matrix: (n_freqs, n_time)
        n_freqs, n_time = cwt_matrices[0].shape
        block_size = n_freqs // self.p_blocks
        
        node_features = []
        
        # 3. Partition into P blocks
        for i in range(self.p_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < self.p_blocks - 1 else n_freqs
            
            # For this block, gather features from ALL channels
            block_feats = []
            for cwt_mat in cwt_matrices:
                block = cwt_mat[start_idx:end_idx, :]
                flat_block = block.flatten()
                block_feats.append(flat_block)
            
            # Concatenate channel features for this node
            # Node feature vector = [Channel1_Block_i, Channel2_Block_i, ...]
            combined_block = np.concatenate(block_feats)
            node_features.append(combined_block)
            
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # 4. Dynamic Edge Construction (Cosine Similarity)
        # Calculate similarity between nodes (blocks)
        x_norm = F.normalize(x, p=2, dim=1)
        cos_sim = torch.mm(x_norm, x_norm.t())
        cos_dist = 1 - cos_sim
        
        # Connect if distance <= kappa
        src, dst = torch.where(cos_dist <= self.kappa)
        edge_index = torch.stack([src, dst], dim=0)
        
        # 5. Prepare Output
        y = torch.tensor([label], dtype=torch.long) if label is not None else None
        
        return Data(x=x, edge_index=edge_index, y=y)
