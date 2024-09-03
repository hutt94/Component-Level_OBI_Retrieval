# -*- codeing = utf-8 -*-
# @Time : 2024/1/20 16:48
# @Author : 李昌杏
# @File : network.py
# @Software : PyCharm
import timm
import torch
import torch.nn as nn
from timm.models import VisionTransformer

class Encoder(nn.Module):
    def __init__(self, num_classes, feature_dim=768, encoder_backbone='vit_base_patch16_224',
                 checkpoint_path='vit.npz'):
        super().__init__()

        self.num_classes = num_classes
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=False,
                                                            checkpoint_path=checkpoint_path)
        self.encoder.embed_dim=feature_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )
        self.encoder.embed_dim=feature_dim

    def embedding(self, photo):
        x = self.encoder.patch_embed(photo)

        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)

        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x[:, 0]

    def classify(self, features):
        return self.mlp_head(features)

    def forward(self, photo):  # class_logits,representation,cls_token,
        x = self.embedding(photo)
        return self.classify(x),x

    def for_visual_attention_map(self,photo):
        x = self.encoder.patch_embed(photo)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x
