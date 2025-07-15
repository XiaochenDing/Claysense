import os
import random
import pandas as pd
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reshape_transform import vit_reshape_transform

# ======== 1. Load DINOv2 pretrained model (ViT) ========
model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
model.eval()

# ======== 2. Image transform ========
transform = Compose([
    Resize(224),                
    CenterCrop((224, 224)),     
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# ======== 3. Load image paths from CSV ========
csv_path = r"Claysense\Dataset\Grad_CAM\GradCAM_labeled.csv"
img_folder = r"Claysense\Dataset\Grad_CAM\dataset_filtered"
df = pd.read_csv(csv_path)
image_paths = df["img_path"].tolist()

# ======== 4. Pick random 10 images ========
sample_paths = random.sample(image_paths, 10)
output_dir = r"Claysense\Dataset\Grad_CAM\gradcam_dinov2_viz"
os.makedirs(output_dir, exist_ok=True)

# ======== 5. Grad-CAM config ========
target_layers = [model.blocks[-1].norm1]  # Last attention block
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=vit_reshape_transform)

for i, rel_path in enumerate(sample_paths):
    abs_path = os.path.join(img_folder, os.path.basename(rel_path))

    image = Image.open(abs_path).convert("RGB")
    
    # Center crop 224x224
    cropped_image = transforms.CenterCrop((224, 224))(image)
    
    # Prepare input tensor
    input_tensor = transform(cropped_image).unsqueeze(0)
    
    # Convert to float for Grad-CAM overlay
    rgb_img = np.array(cropped_image) / 255.0

    # Run Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # Only 1 image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    out_path = os.path.join(output_dir, f"gradcam_img{i}.jpg")
    Image.fromarray(visualization).save(out_path)
    print(f"✅ Saved Grad-CAM visualization to: {out_path}")

