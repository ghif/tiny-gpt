# DATADIR = "/Users/mghifary/Work/Code/AI/data"
DATADIR = "data"
DEVICE = "mps"

# -- Hyperparameters ---
BATCH_SIZE = 64
BLOCK_SIZE = 256

LEARNING_RATE = 3e-4
MAX_ITERS = 5000

EVAL_ITERS = 5
EVAL_INTERVAL = 20

N_EMBED = 384
NUM_HEADS = 6
N_LAYER = 6
# ----------------------

CHECKPOINT_DIR = "models"
MODEL_NAME = "transformer_chairilanwar_v7"