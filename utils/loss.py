import torch
import torch.nn as nn
import torch.nn.functional as F

class MGCNLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, lambda_cor=0.1):
        super(MGCNLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_cor = lambda_cor
        self.ce_loss = nn.CrossEntropyLoss()

    def js_divergence(self, p_logits, q_logits):
       
        p = F.softmax(p_logits, dim=0)
        q = F.softmax(q_logits, dim=0)
        
        m = 0.5 * (p + q)
        
        kl_p_m = F.kl_div(F.log_softmax(p_logits, dim=0), m, reduction='batchmean', log_target=False)
        kl_q_m = F.kl_div(F.log_softmax(q_logits, dim=0), m, reduction='batchmean', log_target=False)
        
        return 0.5 * (kl_p_m + kl_q_m)

    def forward(self, logits_pos, probs_pos, probs_neg, score_pos, score_neg, targets, batch):
         
        l_train = self.ce_loss(logits_pos, targets)
        
        
        js_loss = 0.0
        mse_loss = 0.0
        
        unique_batches = torch.unique(batch)
        num_graphs = len(unique_batches)
        
        for b_id in unique_batches:
            
            mask = (batch == b_id)
            s_p = score_pos[mask]
            s_n = score_neg[mask]
            
            
            eps = 1e-8
            s_p_dist = s_p / (s_p.sum() + eps)
            s_n_dist = s_n / (s_n.sum() + eps)
            
            
            m = 0.5 * (s_p_dist + s_n_dist)
            # KL(P||M) = sum(P * log(P/M))
            kl_p = torch.sum(s_p_dist * torch.log((s_p_dist + eps) / (m + eps)))
            kl_q = torch.sum(s_n_dist * torch.log((s_n_dist + eps) / (m + eps)))
            js = 0.5 * (kl_p + kl_q)
            
            js_loss += js
        
        js_loss = js_loss / num_graphs
        
        mse_loss = F.mse_loss(probs_pos, 1.0 - probs_neg)
        
        l_cor = self.alpha * js_loss + self.beta * mse_loss
        
        l_total = l_train + self.lambda_cor * l_cor
        
        return l_total, l_train, l_cor
