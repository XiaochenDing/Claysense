import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torchvision import transforms
import pandas as pd
from matplotlib import pyplot as plt


DATE = datetime.now().strftime("%d%m%Y")

DATA_DIR = r"Claysense\Dataset"
print(f"DATA_DIR: {DATA_DIR}")

DATASET_NAME = "model1_balanced_DINOv2"
DATA_CSV = os.path.join(
    DATA_DIR,
    "balance_labeled.csv",#"train1.csv" #"final_dataset_full_filtered.csv"
)
    
DATASET_MEAN = [0.3279, 0.2797, 0.3000]
DATASET_STD = [0.1836, 0.1545, 0.1554]
INITIAL_LR = 0.001
BATCH_SIZE = 32 
MAX_EPOCHS = 50 
NUM_NODES = 1
NUM_GPUS = 1
ACCELERATOR = "gpu"

def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.2915257, 0.27048784, 0.14393276],
            [0.2915257, 0.27048784, 0.14393276],
        )
    ],
)