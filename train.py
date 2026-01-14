import torch
import numpy as np
from torch.utils.data import DataLoader
from data.builder import GraphBuilder
from data.dataset import LithologyDataset, collate_graph_batch
from models.mgcn import MGCN_CL
from utils.loss import MGCNLoss
import os

def main():
    print("Initializing MGCN-CL Framework...")

    # --- 1. Configuration ---
    # Hyperparameters
    P_BLOCKS = 8        
    KAPPA = 0.6          
    BATCH_SIZE = 4
    HIDDEN_DIM = 64
    POOLING_RATIO = 0.5
    
    # Dataset Config
    CSV_FILE = 'test_dataset.csv'
    WINDOW_SIZE = 64     
    STRIDE = 32          
    FEATURE_COLS = ['DEN', 'GR', 'CNL', 'AC', 'RT', 'RXO']   
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: Dataset file '{CSV_FILE}' not found.")
        return
 
    print(f"Loading data from {CSV_FILE}...")
    dataset = LithologyDataset(csv_file=CSV_FILE, 
                               window_size=WINDOW_SIZE, 
                               stride=STRIDE, 
                               feature_cols=FEATURE_COLS)
    
    if len(dataset) == 0:
        print("Error: No samples created. Check dataset size vs window size.")
        return

    builder = GraphBuilder(p_blocks=P_BLOCKS, kappa=KAPPA)
     
    collate_fn = lambda x: collate_graph_batch(x, builder)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    
    sample_signal, _ = dataset[0]
    sample_graph = builder.build_graph(sample_signal, 0)
    INPUT_DIM = sample_graph.x.shape[1]
    NUM_CLASSES = dataset.num_classes
    
    print(f"Node Feature Dimension: {INPUT_DIM}")
    print(f"Number of Classes: {NUM_CLASSES}")
    
    
    model = MGCN_CL(input_dim=INPUT_DIM, 
                    hidden_dim=HIDDEN_DIM, 
                    num_classes=NUM_CLASSES, 
                    pooling_ratio=POOLING_RATIO)
    
     
    
    criterion = MGCNLoss(alpha=1.0, beta=1.0, lambda_cor=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    total_correct = 0
    total_samples = 0

    print(f"{'Batch':<8} | {'Loss':<10} | {'Train Loss':<10} | {'Cor Loss':<10} | {'Accuracy':<10}")
    print("-" * 60)

    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward Pass
        logits_pos, probs_pos, probs_neg, score_pos, score_neg, batch_out = model(batch_data)
        
        # Calculate Loss
        targets = batch_data.y.long()  
        loss, l_train, l_cor = criterion(logits_pos, probs_pos, probs_neg, score_pos, score_neg, targets, batch_data.batch)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        # Calculate Accuracy
        preds = torch.argmax(probs_pos, dim=1)
        correct = (preds == targets).sum().item()
        batch_acc = correct / targets.size(0)
        
        total_correct += correct
        total_samples += targets.size(0)

        print(f"{batch_idx+1:<8} | {loss.item():<10.4f} | {l_train.item():<10.4f} | {l_cor.item():<10.4f} | {batch_acc:.2%}")
        
        # Stop after a few batches for demo purposes
        if batch_idx >= 5:
            print("-" * 60)
          
            print(f"Average Accuracy: {total_correct / total_samples:.2%}")
            break
            
    print("Execution successful.")

if __name__ == "__main__":
    main()
