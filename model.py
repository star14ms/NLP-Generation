import torch
import torch.nn as nn
import torch.nn.functional as F


class GT(nn.Module):
    def __init__(self, config):
        super(GT, self).__init__()

        self.config = config

        self.emb = nn.Embedding()
        self.tfd = nn.TransformerDecoderLayer()