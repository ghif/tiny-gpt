# DATADIR = "/Users/mghifary/Work/Code/AI/data"
DATADIR = "data"
DEVICE = "mps"

# -- Hyperparameters ---
BATCH_SIZE = 32
BLOCK_SIZE = 8

LEARNING_RATE = 3e-4
MAX_ITERS = 5000

EVAL_ITERS = 5
# EVAL_ITERS = 200
EVAL_INTERVAL = 20

N_EMBED = 64
NUM_HEADS = 1
N_LAYER = 1

# ----------------------

CHECKPOINT_DIR = "models"
MODEL_NAME = "bigram_chairilanwar_v1"