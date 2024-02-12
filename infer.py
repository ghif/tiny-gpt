import os

import torch
import models as M
import config_v7 as config
# import constants_v1 as const
import data_processor as dp

device = "mps"
datapath = os.path.join(config.DATADIR, "chairilanwar.txt")

chproc = dp.CharProcessor(datapath)

model = M.Transformer(
    chproc.vocab_size, 
    config.BLOCK_SIZE, 
    config.N_EMBED, 
    config.N_LAYER,
    config.NUM_HEADS,
    device=device
)

# Load model
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}.pth")
model.load_state_dict(torch.load(checkpoint_path))
print(f"Model weight loaded: {checkpoint_path}")

print(f"Count parameters: {model.count_parameters()}")

model = model.to(device)

prompt = "taman hati"
input_idx = torch.tensor([chproc.encode(prompt)], dtype=torch.long)
input_str = chproc.decode(input_idx[0].tolist())
print(f"input_str: {input_str}")    

pred_idx = model.generate(input_idx, 500)
pred_str = chproc.decode(pred_idx[0].tolist())

print(f"pred_str: {pred_str}")