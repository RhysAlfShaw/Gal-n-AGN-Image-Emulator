import torch

DATA_PATH = "../Synthetic_Images/cutouts"
SAVE_DIR = "./outputs_diffusers"

BATCH_SIZE = 98
TRAIN_SPLIT = 0.8
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4  # too high and the model will easily jump to a bad local minimum, too low and the model will take forever to converge.
SAMPLE = None  # for inital testin as model takes along time to train.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
