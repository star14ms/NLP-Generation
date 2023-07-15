import torch
import torch.nn as nn
import torch.nn.functional as F


class GT(nn.Module):
    def __init__(self, config):
        super(GT, self).__init__()

        self.config = config

        self.emb = nn.Embedding()
        self.tfd = nn.TransformerDecoderLayer()


# basic transformer
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_emb = nn.Embedding(config.max_len, config.emb_size)
        self.tfd = nn.TransformerDecoderLayer(config.emb_size, config.nhead, config.dim_feedforward, config.dropout)
        self.fc = nn.Linear(config.emb_size, config.vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.shape

        # [batch_size, seq_len, emb_size]
        x = self.emb(x) + self.pos_emb(torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.config.device))
        # [batch_size, seq_len, emb_size]
        x = self.tfd(x, x)
        # [batch_size, seq_len, vocab_size]
        x = self.fc(x)

        return x
    
    def generate(self, x, max_len=20):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.shape

        # [batch_size, seq_len, emb_size]
        x = self.emb(x) + self.pos_emb(torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.config.device))
        # [batch_size, seq_len, emb_size]
        x = self.tfd(x, x)
        # [batch_size, seq_len, vocab_size]
        x = self.fc(x)

        return x
