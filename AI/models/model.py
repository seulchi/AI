from typing import Sequence

import torch
import torch.nn as nn


from .dpt import DPT


class ModelNet(nn.Module):
    """
    Architecture:
    - Backbone: DinoV2 feature extractor
    - Head: DPT for depth prediction

    Returns:
        depth: Predicted depth map (B, 1, H, W)
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        backbone: nn.Module,
        intermediate_layer_idx: Sequence[int],
        dpt_features: int = 256,
        dpt_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        dpt_output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.intermediate_layer_idx = list(intermediate_layer_idx)
        self.dpt = DPT(
            in_channels=backbone.embed_dim,
            output_dim=dpt_output_dim,
            patch_size=self.PATCH_SIZE,
            features=dpt_features,
            out_channels=dpt_out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        _, _, H, W = x.shape
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layer_idx,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        # features: [(patch_tokens, cls_token), ...]
        depth = self.dpt(features, H, W)  # (B, 1, H, W)
        return depth
