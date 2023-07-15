import torch

from data import YTCommentDataset
from config import get_default_config
from model import Transformer
from rich import print


def generate(model, x, max_len=20, end_token_id=3):
    model.eval()

    with torch.no_grad():
        for _ in range(max_len):
            y = model(x)
            y = y[:, -1, :]
            _, next_token = torch.max(y, dim=-1)
            x = torch.cat([x, next_token.unsqueeze(1)], dim=-1)
            if next_token.item() == end_token_id:
                break
            
    return x


if __name__ == '__main__':
    data_file = 'data/yt_cmts_230624_en.txt'
    tokenizer_file = 'data/tokenizer.json'
    model_file = 'model.pt'

    dataset = YTCommentDataset(data_file, tokenizer_file)

    # print(dataset.tokenizer.get_vocab_size(), dataset.get_max_seq_len())

    config = get_default_config(
        dataset.tokenizer.get_vocab_size(), 
        dataset.get_max_seq_len()
    )
    model = Transformer(config)
    model.load_state_dict(torch.load(model_file))

    while True:
        input_text = input('Input: ')

        x = torch.tensor(dataset.tokenizer.encode(input_text).ids).unsqueeze(0)
        y = generate(model, x)

        output_text = dataset.decode(y[0])
        
        print('Output text:', output_text)