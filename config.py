import json
import torch

""" configuration json을 읽어들이는 class """
class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
        

def get_default_config(vocab_size, max_len):
    return Config(
        vocab_size=vocab_size,
        emb_size=512,
        max_len=max_len,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    )