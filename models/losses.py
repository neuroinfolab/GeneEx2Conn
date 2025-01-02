import torch
import torch.nn as nn

class BilinearLoss(nn.Module):
    """MSE loss with optional L1/L2 regularization for bilinear models."""
    def __init__(self, regularization='l1', lambda_reg=1.0):
        super().__init__()
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() if regularization == 'l1' else None
        
    def forward(self, predictions, targets, model):
        mse_loss = self.mse(predictions, targets)

        if self.lambda_reg > 0: 
            weights = [model.linear.weight, model.linear2.weight]
            
            if self.regularization == 'l1':
                reg_loss = sum(self.l1(w, torch.zeros_like(w)) for w in weights)
            elif self.regularization == 'l2':
                reg_loss = sum(torch.sum(w ** 2) for w in weights)
            
            return mse_loss + self.lambda_reg * reg_loss
        
        return mse_loss