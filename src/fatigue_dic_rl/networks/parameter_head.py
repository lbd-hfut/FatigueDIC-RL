"""Dimensionless bounded parameter corrections, with explicit before/after."""
import torch
from torch import nn


class ParameterHead(nn.Module):
    def __init__(self,hidden,parameter_count):
        super().__init__();self.net=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,parameter_count))

    def forward(self,hidden,previous):
        # Smooth bounded update. The delta is in the SAME normalized coordinates.
        updated=torch.tanh(self.net(hidden));delta=updated-previous
        return previous+delta,delta
