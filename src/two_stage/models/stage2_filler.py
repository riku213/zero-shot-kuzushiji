from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        d_model: int = 256,
        layers: int = 4,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.pretrained_backbone: nn.Module | None = None
        if pretrained:
            if patch_size != 16:
                raise ValueError("ImageNet ViT-B/16 weights require patch_size=16.")
            from torchvision import models

            self.pretrained_backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.pretrained_backbone.heads = nn.Identity()
            self.output_projection = (
                nn.Identity() if d_model == 768 else nn.Linear(768, d_model)
            )
        else:
            self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
            patch_count = (image_size // patch_size) ** 2
            self.position_embedding = nn.Parameter(torch.empty(1, patch_count, d_model))
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
            self.output_projection = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.pretrained_backbone is not None:
            backbone = self.pretrained_backbone
            patches = backbone.conv_proj(images).flatten(2).transpose(1, 2)
            encoder = backbone.encoder
            class_token = backbone.class_token.expand(images.shape[0], -1, -1)
            pretrained_position = encoder.pos_embedding
            class_position = pretrained_position[:, :1]
            patch_position = pretrained_position[:, 1:]
            pretrained_side = int(patch_position.shape[1] ** 0.5)
            target_side = self.image_size // self.patch_size
            if pretrained_side != target_side:
                patch_position = patch_position.reshape(1, pretrained_side, pretrained_side, 768)
                patch_position = patch_position.permute(0, 3, 1, 2)
                patch_position = F.interpolate(
                    patch_position, size=(target_side, target_side), mode="bicubic", align_corners=False
                )
                patch_position = patch_position.permute(0, 2, 3, 1).reshape(1, target_side**2, 768)
            sequence = torch.cat((class_token, patches), dim=1)
            sequence = sequence + torch.cat((class_position, patch_position), dim=1)
            return self.output_projection(encoder(sequence)[:, 1:])
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        if patches.shape[1] != self.position_embedding.shape[1]:
            raise ValueError(
                f"Expected {self.position_embedding.shape[1]} image patches, got {patches.shape[1]}."
            )
        return self.encoder(patches + self.position_embedding)


class Stage2SlotFiller(nn.Module):
    """ViT image encoder and bidirectional in-place skeleton decoder."""

    def __init__(
        self,
        input_vocabulary_size: int,
        component_vocabulary_size: int,
        max_skeleton_length: int,
        pad_id: int,
        image_size: int = 128,
        patch_size: int = 16,
        d_model: int = 256,
        encoder_layers: int = 4,
        decoder_layers: int = 3,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if component_vocabulary_size < 1:
            raise ValueError("Stage 2 requires a non-empty leaf-set output vocabulary.")
        self.pad_id = pad_id
        self.image_encoder = VisionTransformerEncoder(
            image_size,
            patch_size,
            d_model,
            encoder_layers,
            attention_heads,
            feedforward_dim,
            dropout,
            pretrained,
        )
        self.skeleton_embedding = nn.Embedding(input_vocabulary_size, d_model, padding_idx=pad_id)
        self.skeleton_position_embedding = nn.Parameter(torch.empty(1, max_skeleton_length, d_model))
        nn.init.trunc_normal_(self.skeleton_position_embedding, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers, norm=nn.LayerNorm(d_model))
        self.output_projection = nn.Linear(d_model, component_vocabulary_size)

    def forward(
        self,
        images: torch.Tensor,
        skeleton_input_ids: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skeleton_input_ids.shape[1] > self.skeleton_position_embedding.shape[1]:
            raise ValueError("Skeleton sequence exceeds the configured maximum length.")
        memory = self.image_encoder(images)
        target = self.skeleton_embedding(skeleton_input_ids)
        target = target + self.skeleton_position_embedding[:, : skeleton_input_ids.shape[1]]
        if target_padding_mask is None:
            target_padding_mask = skeleton_input_ids.eq(self.pad_id)
        decoded = self.decoder(
            target,
            memory,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=None,
        )
        return self.output_projection(decoded)


def slot_only_cross_entropy(
    logits: torch.Tensor,
    target_positions: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    slot_logits = logits[rows, target_positions]
    return nn.functional.cross_entropy(slot_logits, target_ids)