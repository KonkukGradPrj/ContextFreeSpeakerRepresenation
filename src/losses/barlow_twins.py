import torch
import torch.nn.functional as F

# matrix에서 diagonal이 아닌 값이 flatten되어 나옴
def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()

def barlow_twins_loss(cross_corr_matrix,is_positive=True):

    scale_loss = float(1/32)
    lambd = float(3.9e-3)

    on_diag = torch.diagonal(cross_corr_matrix).add_(-1).pow_(2).mul(scale_loss)
    off_diag = off_diagonal(cross_corr_matrix).pow_(2).sum().mul(scale_loss)

    if is_positive:
        loss = on_diag + lambd*off_diag
    else:
        loss = off_diag + lambd*on_diag

    return loss
