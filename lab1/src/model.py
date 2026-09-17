import torch
import torch.nn as nn
import torch.nn.functional as F


class autoencoderMLP4Layer(nn.Module):

    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = N_input // 2
        # ---------------------------------
        # Your code here ...
        # ---------------------------------

    def encode(self, X):
        # ---------------------------------
        # Your code here ...
        # ---------------------------------

    def decode(self, X):
        # ---------------------------------
        # Your code here ...
        # ---------------------------------

    def forward(self, X):
        return self.decode(self.encode(X))
