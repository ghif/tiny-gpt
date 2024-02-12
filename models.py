import torch
import torch.nn as nn
from torch.nn import functional as F
import os

torch.manual_seed(1337)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        B, T, head_size = k.shape

        # compute attention scores ("affinities")
        A = q @ k.transpose(-2, -1) * head_size ** -0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        A = A.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        A = F.softmax(A, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        H = A @ v # (B, T, head_size)
        return H

class MultiHeadAttention(nn.Module):
    """ Multiheads of self-attention in parallel """

    def __init__(self, n_embed, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class Feedforward(nn.Module):
    """ A simple MLP with non-linear activation"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # projection layer
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ A Transformer block """

    def __init__(self, n_embed, num_heads, block_size):
        super().__init__()
        head_size = n_embed // num_heads
        self.mha = MultiHeadAttention(n_embed, num_heads, head_size, block_size)
        self.ffwd = Feedforward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleBigram(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, device="cpu"):
        """
        Args:
            vocab_size (int): size of the vocabulary
            block_size (int): number of time steps in a sequence
            n_embed (int): dimension of the embedding
            device (str): device to run the model
        """
        super().__init__()

        self.device = device
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        """
        Args:
            idx (torch.tensor): (B, T) array of indices in the current context
            targets (torch.tensor, default None): (B, T) array of indices for the next token for computing the loss function
        
        Returns:
            logits (torch.tensor): (B, T, vocab_size) array of logits
            loss (torch.tensor): scalar loss value if targets is not None, otherwise None
        """
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        # x = tok_emb + pos_emb # (B, T, C)
        x = tok_emb
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            pred = logits.view(B*T, C)
            groundtruth = targets.view(B*T)        
            loss = F.cross_entropy(pred, groundtruth)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens/characters given an input

        Args:
            idx (torch.tensor): (B, T) array of indices in the current context
            max_new_tokens (int): maximum number of new tokens to generate
        Returns:
            idx (torch.tensor): (B, T+max_new_tokens) array of indices in the current context
        """
        self.eval()

        idx = idx.to(self.device)
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Predict
            # logits, loss = self(idx)
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        self.train()    
        return idx
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def save(self, checkpoint_dir, model_name):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
        torch.save(self.state_dict(), model_path)
        
        print(f"Saved PyTorch Model State to {model_path}")

    
class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_layer, num_heads, device="cpu"):
        super().__init__()

        self.device = device
        self.block_size = block_size

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # self.sa_head = Head(n_embed, head_size) # one head of self-attention

        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size) # language model head

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)

        # x = self.sa_head(x) # (B, T, head_size)
        # x = self.ffwd(x) # (B, T, vocab_size)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            pred = logits.view(B*T, C)
            groundtruth = targets.view(B*T)        
            loss = F.cross_entropy(pred, groundtruth)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()

        idx = idx.to(self.device)
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # print(f"idx shape: {idx.shape}")
            idx_cond = idx[:, -self.block_size:]
            # print(f"idx_cond shape: {idx_cond.shape}")
            # Predict
            # logits, loss = self(idx)
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        self.train()    
        return idx
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def save(self, checkpoint_dir, model_name):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
        torch.save(self.state_dict(), model_path)
        
        print(f"Saved PyTorch Model State to {model_path}")