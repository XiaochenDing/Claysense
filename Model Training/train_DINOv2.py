import torch
import os
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from train_config import *
from data.data_module_wholeworkflow import ParametersDataModule
from model.network_module_DINOv2 import DINO2ResAttClassifier

def set_seed(seed):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed",   default=1234,   type=int, help="Set seed")
    parser.add_argument("-e", "--epochs", default=MAX_EPOCHS, type=int, help="Num epochs")
    args = parser.parse_args()

    set_seed(args.seed)
    logs_dir = os.path.join("logs", f"logs-{DATE}", str(args.seed))
    make_dirs(logs_dir); make_dirs(os.path.join(logs_dir, "default"))

    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
    ckpt_cb   = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join("checkpoints", DATE, str(args.seed)),
        filename=f"DINO2ResAtt-{DATASET_NAME}-{DATE}-{{epoch:02d}}-{{val_loss:.2f}}-{{val_acc:.2f}}",
        save_top_k=3, mode="min"
    )

    # 1) DINO2ResAttClassifier
    model = DINO2ResAttClassifier(num_classes=3, lr=INITIAL_LR)

    # 2) Datamodule
    data = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=DATA_CSV,
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
    )

    # 3) Trainer
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,               
            precision="16-mixed",    
            max_epochs=args.epochs,
            callbacks=[ckpt_cb],
            logger=tb_logger,
        )
    else:
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,               
            precision=32,            
            max_epochs=args.epochs,
            callbacks=[ckpt_cb],
            logger=tb_logger,
        )

    trainer.fit(model, data, ckpt_path= r"Claysense\checkpoints\19062025\1234\DINO2ResAtt-model1_balanced_DINOv2-19062025-epoch=11-val_loss=0.37-val_acc=0.00.ckpt")

if __name__ == "__main__":
    main()
