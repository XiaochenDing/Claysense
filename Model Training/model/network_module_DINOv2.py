# model/network_module_DINOv2.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Original ResAttNet-56 Second half
from model.residual_attention_network import ResidualAttentionModel_56

# DINOv2
dino_v2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=True)
dino_v2.eval()
for p in dino_v2.parameters():
    p.requires_grad = False


class DINO2ResAttClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, lr=1e-3): #1e-4
        super().__init__()
        self.lr = lr

        # — (A) DINOv2 —
        self.dino = dino_v2
        self.conv_reduce = nn.Conv2d(384, 1024, 1, bias=False)

        # — (B) ResAttNet-56 Second half —
        full = ResidualAttentionModel_56()
        self.res4 = full.residual_block4
        self.res5 = full.residual_block5
        self.res6 = full.residual_block6
        self.bn_relu      = nn.Sequential(nn.BatchNorm2d(2048), nn.ReLU(inplace=True))
        self.adaptive_pool= nn.AdaptiveAvgPool2d((1,1))

        # Two-end classifier
        self.fc_ex = nn.Linear(2048, num_classes)
        self.fc_ov = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 1) Extract all tokens using DINOv2
        feat_dict = self.dino.forward_features(x)
        # 2) Get patch tokens from dict
        patches = feat_dict["x_norm_patchtokens"]      # [B, N, 384]
        B, N, C = patches.shape
        H = W = int(math.sqrt(N))
        spatial = patches.transpose(1, 2).reshape(B, C, H, W)  # [B,384,H,W]
        # 3) Projection to 1024 channels
        x = self.conv_reduce(spatial)
        # 4) ResAttNet-56 Second half
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.bn_relu(x)
        x = self.adaptive_pool(x)
        x = x.view(B, -1)  # [B,2048]

        # 5) two-head outputs
        out_ex = self.fc_ex(x)
        out_ov = self.fc_ov(x)
        return out_ex, out_ov


    def configure_optimizers(self):
        opt   = AdamW(self.parameters(), lr=self.lr)
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=3)
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}

    def _step(self, batch, name):
        x, y    = batch
        y_ex, y_ov = y[:,0], y[:,1]
        p_ex, p_ov = self(x)
        loss = F.cross_entropy(p_ex, y_ex) + F.cross_entropy(p_ov, y_ov)

        self.log(f"{name}_loss", loss,    prog_bar=(name=="train"), on_epoch=True)
        self.log(f"{name}_acc_ex", (p_ex.argmax(1)==y_ex).float().mean(), on_epoch=True)
        self.log(f"{name}_acc_ov", (p_ov.argmax(1)==y_ov).float().mean(), on_epoch=True)
        comb_acc = ((p_ex.argmax(1)==y_ex)&(p_ov.argmax(1)==y_ov)).float().mean()
        self.log(f"{name}_acc_comb", comb_acc, prog_bar=(name=="train"), on_epoch=True)
        return loss

    def training_step(self,    b,i): return self._step(b,"train")
    def validation_step(self,  b,i): return self._step(b,"val")
    def test_step(self,        b,i): return self._step(b,"test")
