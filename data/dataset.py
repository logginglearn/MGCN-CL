import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from data.builder import GraphBuilder

class LithologyDataset(Dataset):
   
    def __init__(self, csv_file, window_size=64, stride=32, feature_cols=['GR']):
        """
        Args:
            csv_file (str): Path to CSV.
            window_size (int): Size of the signal window for CWT.
            stride (int): Step size for sliding window.
            feature_cols (str or list): The column(s) to use as the signal.
        """
        self.df = pd.read_csv(csv_file)
        
        # Sort by depth to ensure sequentiality
        if 'DEPTH' in self.df.columns:
            self.df = self.df.sort_values('DEPTH')
        
        if isinstance(feature_cols, str):
            feature_cols = [feature_cols]
            
        # Select columns and ensure numeric type
        self.signal_data = self.df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        # --- Label Handling ---
        if 'classification' in self.df.columns:
            self.labels = self.df['classification'].values
        elif 'LITHOLOGY' in self.df.columns:
            # Fallback if classification col missing, encode string labels
            raw_labels = self.df['LITHOLOGY'].values
            unique_classes = sorted(list(set(raw_labels)))
            class_map = {cls: i for i, cls in enumerate(unique_classes)}
            self.labels = np.array([class_map[l] for l in raw_labels])
        else:
            raise ValueError("No valid label column found (checked 'classification', 'LITHOLOGY')")
        
        self.window_size = window_size
        self.stride = stride
        
        # Create windows
        self.samples = []
        self.sample_labels = []
        
        num_points = len(self.signal_data)
        for i in range(0, num_points - window_size + 1, stride):
            # signal shape: (window_size, num_channels)
            window_signal = self.signal_data[i : i + window_size]
            
            # Use the label of the center point of the window
            center_idx = i + window_size // 2
            window_label = self.labels[center_idx]
            
            self.samples.append(window_signal)
            self.sample_labels.append(window_label)
            
        self.samples = np.array(self.samples)
        self.sample_labels = np.array(self.sample_labels)
        
        # Mapping labels to 0-N if not already
        self.unique_labels = np.unique(self.sample_labels)
        self.num_classes = len(np.unique(self.labels)) # Use total dataset classes
        
        print(f"Dataset loaded: {len(self.samples)} windows created from {num_points} points.")
        print(f"Input shape: {self.samples.shape}") # (N, window_size, num_channels)
        print(f"Classes found in windows: {self.unique_labels}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # returns (window_size, num_channels), scalar_label
        return self.samples[idx], self.sample_labels[idx]

def collate_graph_batch(batch_list, graph_builder):
    """
    Custom collate function to convert a batch of (signal, label) to a PyG Batch object.
    """
    data_list = []
    for signal, label in batch_list:
        data = graph_builder.build_graph(signal, label)
        data_list.append(data)
    
    return Batch.from_data_list(data_list)
