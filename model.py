import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / torch.FloatTensor([d_model]))
        return position * angles
  
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=torch.arange(position).unsqueeze(1),
            i=torch.arange(d_model).unsqueeze(0),
            d_model=d_model)
    
        # apply the sine function to the even indices of the array (2i)
        sines = torch.sin(angle_rads[:, 0::2])
    
        # apply the cosine function to the odd indices of the array (2i+1)
        cosines = torch.cos(angle_rads[:, 1::2])
    
        pos_encoding = torch.zeros(angle_rads.shape)
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines
        pos_encoding = pos_encoding.unsqueeze(0)
    
        return pos_encoding.float()
  
    def forward(self, inputs):
        return inputs + self.pos_encoding[:, :inputs.shape[1], :]


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.emb_size)
        self.pos_emb = PositionalEncoding(config.max_len, config.emb_size)
        self.tfd = nn.TransformerDecoderLayer(config.emb_size, config.nhead, config.dim_feedforward, config.dropout)
        self.fc = nn.Linear(config.emb_size, config.vocab_size)

    def forward(self, x):
        # [batch_size, seq_len, emb_size]
        x = self.pos_emb(self.emb(x))

        # [batch_size, seq_len, emb_size]
        x = self.tfd(x, x)
        # [batch_size, seq_len, vocab_size]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
  
    position = 100 # length of the sequence
    d_model = 64 # dimension of the embedding vector
    sample_pos_encoding = PositionalEncoding(position, d_model)
  
    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
