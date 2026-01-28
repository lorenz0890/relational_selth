import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer

EMBED = 16 #32
LAYERS = 3 #4
HIDDEN = 16 #1024

BATCH = 1024
PATIENCE = 100 # Wont be used if ste to 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_optimiser(model: Module) -> Optimizer:
    return Adam(model.parameters())
