import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence


tokenizer_file = 'data/tokenizer.json'

tokenizer = Tokenizer.from_file(tokenizer_file)
padding_value = tokenizer.token_to_id('[PAD]')


class YTCommentDataset(Dataset):
    def __init__(self, data_file, tokenizer_file):
        super().__init__()

        self.data_file = data_file
        self.tokenizer_file = tokenizer_file

        # load the tokenizer
        self.tokenizer = Tokenizer.from_file(self.tokenizer_file)

        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()

        # encode the line
        encoded = self.tokenizer.encode(f'[SOS] {line} [EOS]')

        # convert to torch tensor
        x = torch.tensor(encoded.ids[:-1])
        y = torch.tensor(encoded.ids[1:])

        return x, y
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def get_max_seq_len(self):
        return max([len(self.tokenizer.encode(line).ids) for line in self.lines])


def collate_fn(batch):
    x, y = zip(*batch)

    x = pad_sequence(x, batch_first=True, padding_value=padding_value)
    y = pad_sequence(y, batch_first=True, padding_value=padding_value)

    return x, y

def get_dataloader(data_file, tokenizer_file, batch_size=4, shuffle=True):
    dataset = YTCommentDataset(data_file, tokenizer_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    return dataloader


if __name__ == '__main__':
    data_file = 'data/yt_cmts_230624_en.txt'
    tokenizer_file = 'data/tokenizer.json'

    tokenizer = Tokenizer.from_file(tokenizer_file)
    padding_value = tokenizer.token_to_id('[PAD]')
    dataset = YTCommentDataset(data_file, tokenizer_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    x, y = dataset[0]

    print(x)
    print(y)

    print(dataset.decode(x))
    print(dataset.decode(y))

    for x, y in dataloader:
        print(x.shape, y.shape)
        break