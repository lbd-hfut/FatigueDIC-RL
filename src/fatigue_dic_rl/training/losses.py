import torch


def parameter_loss(mean,target):return torch.mean((mean-target)**2)


def gaussian_parameter_nll(mean,root,target):
    error=(target-mean).unsqueeze(-1);w=torch.linalg.solve_triangular(root,error,upper=False)
    return (.5*w.squeeze(-1).square().sum(-1)+torch.log(torch.diagonal(root,dim1=-2,dim2=-1)).sum(-1)).mean()
