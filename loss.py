import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    
    def __init__(self, gamma=1.5, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)

        loss = - self.alpha * (1 - pred) ** self.gamma * target * torch.log(pred + 1e-12) - (1 - self.alpha) * pred ** self.gamma * (1 - target) * torch.log(1 - pred + 1e-12)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss