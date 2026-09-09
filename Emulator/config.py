import torch

DATA_PATH = "../Synthetic_Images/cutouts"
SAVE_DIR = "./outputs_diffusers"

BATCH_SIZE = 98  # as much as can fit in GPU memory, for compute efficiency.
TRAIN_SPLIT = 0.8
NUM_EPOCHS = 1000  # depending on size of dataset and batch size. Anywhere from 500-2000 epochs might be needed for convergence.
LEARNING_RATE = 1e-4  # too high and the model will easily jump to a bad local minimum, too low and the model will take forever to converge.
SAMPLE = (
    None  # can help for inital testing, but becareful when increasing the size again
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
