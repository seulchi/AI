from typing import Sequence

import torch
import torch.nn as nn


class ModelNet(nn.Module):
    """
    Architecture:
    - Backbone: DinoV2 feature extractor
    """

    def __init__(
        self,
        backbone: nn.Module,
        intermediate_layer_idx: Sequence[int],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.intermediate_layer_idx = list(intermediate_layer_idx)

    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W)
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layer_idx,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        # features: [(patch_tokens, cls_token), ...]
        return features
