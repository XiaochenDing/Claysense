import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        crop_size=224,
        pre_crop_transform=None,
        post_crop_transform=None,
        regression=False,
        extrusion=False,
        overhang=False,
        per_img_normalisation=False,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform
        self.use_extrusion = extrusion
        self.use_overhang = overhang
        self.per_img_normalisation = per_img_normalisation
        self.targets = []

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        self.targets = []
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.img_path[idx])
        image = Image.open(img_name)


        width, height = image.size
        half = self.crop_size // 2
        center_x, center_y = width // 2, height // 2

        left   = max(center_x - half, 0)
        top    = max(center_y - half, 0)
        right  = min(center_x + half, width)
        bottom = min(center_y + half, height)

        image = image.crop((left, top, right, bottom))

        
        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)

        if self.per_img_normalisation:
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            image = tfms(image)
            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean, std)(image)
        else:
            if self.post_crop_transform:
                image = self.post_crop_transform(image)

        if self.use_extrusion:
            extrusion_class = int(self.dataframe.extrusion_class[idx])
            self.targets.append(extrusion_class)

        if self.use_overhang:
            overhang_class = int(self.dataframe.overhang_class[idx])
            self.targets.append(overhang_class)


        y = torch.tensor(self.targets, dtype=torch.long)
        sample = (image, y)
        return sample
