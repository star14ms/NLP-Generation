import torch

from data import get_dataloader
from config import get_default_config
from model import Transformer

from utils._rich import new_progress


def train_epoch(model, dataloader, optimizer, loss_fn, progress):

    task_id = progress.add_task('Training', total=len(dataloader))

    # for i, (x, y) in enumerate(dataset):
    #     x, y = dataset[0]
    #     x = x.unsqueeze(0) # shape: [1, seq_len]

    for i, (x, y) in enumerate(dataloader):
        y = model(x)
        loss = loss_fn(y.view(-1, y.size(-1)), x.view(-1))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        progress.update(task_id, advance=1)

        if (i + 1) % 100 == 0:
            progress.log(f'step {i+1}, loss: {loss.item()}')
            progress.refresh()

        if (i + 1) % 1000 == 0:
            torch.save(model.state_dict(), 'model.pt')


if __name__ == '__main__':
    data_file = 'data/yt_cmts_230624_en.txt'
    tokenizer_file = 'data/tokenizer.json'

    dataloader = get_dataloader(data_file, tokenizer_file, batch_size=4, shuffle=True)

    config = get_default_config(
        dataloader.dataset.tokenizer.get_vocab_size(), 
        dataloader.dataset.get_max_seq_len()
    )
    model = Transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    with new_progress() as progress:
        train_epoch(model, dataloader, optimizer, loss_fn, progress)
        torch.save(model.state_dict(), 'model.pt')

    breakpoint()