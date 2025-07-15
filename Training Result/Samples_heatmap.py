import os
from model.network_module_DINOv2 import DINO2ResAttClassifier
from PIL import Image
from train_config import *
import time
import os
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning import seed_everything

from data.data_module_wholeworkflow import ParametersDataModule
import pandas as pd
from matplotlib import pyplot as plt




########## predict label from orginal sample file ###################
sample_data = r"Claysense\Dataset\prediction_sample"
model = DINO2ResAttClassifier.load_from_checkpoint(
    checkpoint_path=r"Claysense\checkpoints\19062025\1234\DINO2ResAtt-model1_balanced_DINOv2-19062025-epoch=46-val_loss=0.15-val_acc=0.00.ckpt",
    num_classes=3,
    gpus=1,
)
model.eval()
img_paths = [
    os.path.join(sample_data, img)
    for img in os.listdir(sample_data)
    if os.path.splitext(img)[1] == ".jpg"
]

print("********* Claysense sample predictions *********")
print("Extrusion | Overhang")
print("*********************************************")

########### Plot heatmap\ confusion matrix ##############
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re  

true_labels_param1 = []
pred_labels_param1 = []
true_labels_param2 = []
pred_labels_param2 = []

for img_path in img_paths:
    pil_img = Image.open(img_path)
    x = preprocess(pil_img).unsqueeze(0)
    y_hats = model(x)
    y_hat0, y_hat1 = y_hats

    _, pred_param1 = torch.max(y_hat0, 1)
    _, pred_param2 = torch.max(y_hat1, 1)
    
    pred_labels_param1.append(pred_param1.item())
    pred_labels_param2.append(pred_param2.item())

    img_basename = os.path.basename(img_path)
    match = re.search(r'_label_\[(\d) (\d)\]', img_basename)
    if match:
        true_label_param1 = int(match.group(1))
        true_label_param2 = int(match.group(2))
        
        true_labels_param1.append(true_label_param1)
        true_labels_param2.append(true_label_param2)
    else:
        print(f"Could not parse labels from filename: {img_basename}")


cm_param1 = confusion_matrix(true_labels_param1, pred_labels_param1, labels=[0, 1, 2])
cm_param2 = confusion_matrix(true_labels_param2, pred_labels_param2, labels=[0, 1, 2])

# Plot heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
titles = ["Parameter 1", "Parameter 2"]
cms = [cm_param1, cm_param2]

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, 
                xticklabels=["Low", "Good", "High"], yticklabels=["Low", "Good", "High"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

plt.tight_layout()
plt.show()
