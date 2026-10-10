"""Full Cholesky covariance learned from parameter errors, not MAP covariance."""
import torch
from torch import nn


class UncertaintyHead(nn.Module):
    def __init__(self,hidden,parameter_count):
        super().__init__();self.k=parameter_count;self.net=nn.Linear(hidden,parameter_count*(parameter_count+1)//2)
        ids=torch.tril_indices(parameter_count,parameter_count);self.register_buffer('rows',ids[0]);self.register_buffer('cols',ids[1])

    def forward(self,hidden):
        raw=self.net(hidden);root=raw.new_zeros((*raw.shape[:-1],self.k,self.k));root[...,self.rows,self.cols]=raw
        diagonal=torch.arange(self.k,device=raw.device);root[...,diagonal,diagonal]=torch.nn.functional.softplus(root[...,diagonal,diagonal])+1e-4
        return root@root.transpose(-1,-2),root
