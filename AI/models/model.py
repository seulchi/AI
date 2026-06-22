from typing import Sequence, Tuple

import torch
import torch.nn as nn


from .dpt import DPT


class ModelNet(nn.Module):
    """
    Architecture:
    - Backbone: DinoV2 feature extractor
    - Head: DPT for factored ray prediction

    Returns:
        ray_directions: per-pixel unit vectors (B, 3, H, W)
        ray_depths: Euclidean distance from camera center (B, 1, H, W)
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        backbone: nn.Module,
        intermediate_layer_idx: Sequence[int],
        dpt_features: int = 256,
        dpt_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        dpt_output_dim: int = 4,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        ray_directions, ray_depths = self.dpt(features, H, W)
        return ray_directions, ray_depths
