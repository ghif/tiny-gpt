import os
import json

import torch
import models as M
import config_bigram_v2 as config
from time import process_time
import data_processor as dp

datapath = os.path.join(config.DATADIR, "chairilanwar.txt")

chproc = dp.CharProcessor(datapath)

data = torch.tensor(chproc.encode(chproc.text), dtype=torch.long)
print(data.shape, data.dtype)

# Construct training data
n = len(data)
train_data = data[:n]
val_data = None

# Print pairs of input and target
x = train_data[:config.BLOCK_SIZE]
y = train_data[1:config.BLOCK_SIZE+1]
for t in range(config.BLOCK_SIZE):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, the target is {target}")

torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss(
        model, 
        data, 
        eval_iters=10
    ):
    
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = dp.get_batch(data, batch_size=config.BATCH_SIZE, block_size=config.BLOCK_SIZE)
        xb = xb.to(config.DEVICE)
        yb = yb.to(config.DEVICE)

        logits, loss = model(xb, yb)
        losses[k] = loss.item()

    avg_loss = losses.mean()
    model.train()

    return avg_loss

xb, yb = dp.get_batch(
    train_data,
    batch_size=config.BATCH_SIZE,
    block_size=config.BLOCK_SIZE
)

xb = xb.to(config.DEVICE)
yb = yb.to(config.DEVICE)

model = M.SimpleBigram(
    chproc.vocab_size,
    config.BLOCK_SIZE,
    config.N_EMBED,
    device=config.DEVICE
)

model = model.to(config.DEVICE)
logits, loss = model(xb, yb)

print(f"Count parameters: {model.count_parameters()}")

idx = torch.zeros((1, 1), dtype=torch.long)
pred_idx = model.generate(idx, 1000)
pred_str = chproc.decode(pred_idx[0].tolist())

print(f"pred_idx: {pred_idx}")
print(f"pred_str: {pred_str}")


# Create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

val_loss = 99999
train_loss = 99999

print(f"Using device: {config.DEVICE}")
history = {
    'train_losses': [],
    'train_times': [],
}
elapsed_times = []
for step in range(config.MAX_ITERS):
    # Sample a batch of data
    xb, yb = dp.get_batch(train_data, batch_size=config.BATCH_SIZE, block_size=config.BLOCK_SIZE)
    xb = xb.to(config.DEVICE)
    yb = yb.to(config.DEVICE)

    start_t = process_time()

    # Evaluate the loss
    logits, loss = model(xb, yb)

    # Backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    elapsed_t = process_time() - start_t
    elapsed_times.append(elapsed_t)
    history['train_times'] = elapsed_times

    if step % config.EVAL_INTERVAL == 0:
        
        start_t = process_time()
        train_loss = estimate_loss(model, train_data, eval_iters=config.EVAL_ITERS)
        if val_data is not None:
            val_loss = estimate_loss(model, val_data, eval_iters=config.EVAL_ITERS)

        history['train_losses'].append(train_loss.item())

        val_elapsed_t = process_time() - start_t
        
        print(f"Step-{step+1}/{config.MAX_ITERS} [elapsed time: {elapsed_t:.5f}secs (train), {val_elapsed_t:.5f}secs (val)]: train loss={train_loss:.4f}, validation loss={val_loss:.4f}")

        # Save the model
        model.save(config.CHECKPOINT_DIR, config.MODEL_NAME)

        # Save training history
        history_path = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_hist.json")
        
        json_object = json.dumps(history) # serializing json
        with open(history_path, "w") as outfile:
            outfile.write(json_object) # write to json file

        pred_idx = model.generate(idx, 100)
        pred_str = chproc.decode(pred_idx[0].tolist())
        print(f"\npred_str: {pred_str}\n")

pred_idx = model.generate(idx, 1000)
pred_str = chproc.decode(pred_idx[0].tolist())

print(f"pred_idx: {pred_idx}")
print(f"pred_str: {pred_str}")

print(f"Total elapsed time: {sum(elapsed_times):.5f} secs")