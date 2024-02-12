import os
import torch
import constants as const

class CharProcessor:
    def __init__(self, datapath):
        # Read the data
        with open(datapath, "r", encoding="utf-8") as f:
            self.text = f.read()

        print(f"Length of text: {len(self.text)} characters")

        # The unique characters in the file
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print("".join(self.chars))
        print(self.vocab_size)

        # Create a mapping from characters to integers
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = { i:ch for i, ch in enumerate(self.chars) }
        self.encode = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: "".join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

def get_batch(data, batch_size=4, block_size=8):
    """
    Generate a batch of data of inputs x and target y

    Args:
        data (torch.tensor): (N,) full encoded text in integers
        batch_size (int): number of samples in a batch
        block_size (int): number of time steps in a sequence
    
    Returns:
        x (torch.tensor): (batch_size, block_size) input sequence
        y (torch.tensor): (batch_size, block_size) output sequence
    """
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[t:t+block_size] for t in ix])
    y = torch.stack([data[t+1:t+block_size+1] for t in ix])
    return x, y

if __name__ == "__main__":
    datapath = os.path.join(const.DATADIR, "shakespeare.txt")
    chproc = CharProcessor(datapath)

    data = torch.tensor(chproc.encode(chproc.text), dtype=torch.long)

    # Create a simple data batch
    B = 4
    T = 8
    xb, yb = get_batch(data, batch_size=B, block_size=T)
    print(f"xb: {xb}")
    print(f"yb: {yb}")

    x = xb[0]
    y = yb[0]

    for t in range(T-1):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context}, the target is {target}")


