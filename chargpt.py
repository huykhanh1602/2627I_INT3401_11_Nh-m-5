"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


Modified version of Andrej Karpathy's "mingpt" repo found here: https://github.com/karpathy/minGPT

Nhóm 5 chỉnh sửa so với bản gốc (xem chargpt_goc.py):
  - thêm tham số dòng lệnh (--lr, --block, --embd, --layers, ...) để chạy thí nghiệm mà không phải sửa file
  - tách 90% dữ liệu để huấn luyện, 10% để đo val loss, in train/val loss mỗi --checkpoint vòng
  - dùng lr=learning_rate (bản gốc ghi cứng lr=0.00004 và không dùng biến learning_rate)
  - in cấu hình ở đầu log để log tự mô tả được lần chạy

Ví dụ:
  python -u chargpt.py                                   # cấu hình mặc định (lr 0.0004)
  python -u chargpt.py --lr 0.00004 --tag A              # lr của bản gốc
  python -u chargpt.py --lr 0.001 --tag C
  python -u chargpt.py --block 32 --tag D
  python -u chargpt.py --iters 200 --tag thu             # chạy thử nhanh để đo tốc độ
"""

import argparse
import random
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from gpt_model import Character_GPT



class CharDataset(Dataset):
    def __init__(self, data, block_size=10):
        #Configure block size
        self.block_size = block_size

        #Define possible characters, and create mapping from number to character
        chars = sorted(list(set(data)))
        self.index2char = { i:ch for i,ch in enumerate(chars) }
        self.char2index = { ch:i for i,ch in enumerate(chars) }
        vocab_size = len(chars)
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.char2index[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:],dtype=torch.long)
        return x, y



def train_single_iteration(model, data_iter):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    batch = [t for t in batch]
    x, y = batch

    # forward the model
    loss = model.get_loss(x, y)

    # backprop and update the parameters
    model.zero_grad(set_to_none=True)
    loss.backward()
   
    optimizer.step()
    return loss.item()


def eval_loss(model, dataset, n_batches=20, batch_size=500):
    # average loss on randomly chosen chunks of the held-out data
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            idxs = [random.randrange(len(dataset)) for _ in range(batch_size)]
            xs, ys = zip(*[dataset[k] for k in idxs])
            total += model.get_loss(torch.stack(xs), torch.stack(ys)).item()
    return total / n_batches


def sample_text(model, dataset, prompt, n_chars):
    with torch.no_grad():
        x = torch.tensor([dataset.char2index[s] for s in prompt])[None, ...]
        y = model.generate(x, n_chars)[0]
        return ''.join([dataset.index2char[int(j)] for j in y])

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    parser.add_argument('--iters', type=int, default=10000, help='so vong lap')
    parser.add_argument('--batch', type=int, default=500)
    parser.add_argument('--block', type=int, default=10, help='do dai ngu canh (ky tu)')
    parser.add_argument('--embd', type=int, default=100, help='so chieu embedding (layer_size)')
    parser.add_argument('--layers', type=int, default=6, help='so khoi Transformer')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', type=int, default=100, help='moi bao nhieu vong thi in loss')
    parser.add_argument('--sample_every', type=int, default=1000, help='moi bao nhieu vong thi sinh van ban')
    parser.add_argument('--prompt', type=str, default='Pacman')
    parser.add_argument('--tag', type=str, default='', help='nhan cua lan chay (chi de in ra log)')
    parser.add_argument('--save', type=str, default='', help='duong dan luu trong so, vi du q7_C.pt')
    args = parser.parse_args()

    #Variables to configure
    learning_rate = args.lr
    num_iterations = args.iters
    batch_size = args.batch
    checkpoint = args.checkpoint #How often to print out results of the model during training
    context = args.prompt #Generative prompt
    layer_size = args.embd
    n_layer = args.layers #How many transformer blocks to have
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # construct the training dataset (90% train / 10% validation)
    text = open('input.txt', 'r').read()
    split = int(0.9 * len(text))
    train_dataset = CharDataset(text, args.block)   # builds the character table from the whole text
    val_dataset = CharDataset(text, args.block)
    train_dataset.data = text[:split]
    val_dataset.data = text[split:]

    missing = [c for c in context if c not in train_dataset.char2index]
    if missing:
        raise SystemExit("Prompt co ky tu khong co trong input.txt: " + repr(''.join(missing)))

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        batch_size=batch_size,
        num_workers=1,
    )

    train_iterations = iter(train_loader)

    #set up model and optimizer
    model = Character_GPT(train_dataset.block_size, n_embd=layer_size, n_layer=n_layer, vocab_size=train_dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print("CONFIG: tag=%r lr=%g iters=%d batch=%d block=%d embd=%d layers=%d seed=%d params=%d vocab=%d"
          % (args.tag, learning_rate, num_iterations, batch_size, args.block, layer_size, n_layer,
             args.seed, n_params, train_dataset.vocab_size), flush=True)

    t0 = time.time()
    best_val = float('inf')
    for i in range(num_iterations):
        loss = train_single_iteration(model, train_iterations)

        if i % checkpoint == 0:
            val_loss = eval_loss(model, val_dataset)
            best_val = min(best_val, val_loss)
            print(f"Iteration {i}: train loss {loss:.4f}, val loss {val_loss:.4f} ({time.time() - t0:.0f}s)", flush=True)

        if i % args.sample_every == 0:
            print("Prompt: " + context)
            print("Generated result: ")
            print(sample_text(model, train_dataset, context, 500) + "\n", flush=True)

    final_val = eval_loss(model, val_dataset, n_batches=100)
    print(f"FINAL: val loss {final_val:.4f} (trung binh 100 batch), val loss tot nhat luc kiem tra {best_val:.4f}, "
          f"thoi gian {time.time() - t0:.0f}s", flush=True)
    print("FINAL SAMPLE:")
    print(sample_text(model, train_dataset, context, 800), flush=True)

    if args.save:
        torch.save(model.state_dict(), args.save)
        print("Da luu trong so vao " + args.save)