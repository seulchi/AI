from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


from .dpt import DPT


class ModelNet(nn.Module):
    """
    Architecture:
    - Backbone: DinoV2 feature extractor
    - Head: DPT for factored ray prediction

    Args:
        views (List[torch.Tensor]): List of input views' images. (B, 3, H, W)

    Returns:
        ray_directions (Tuple[torch.Tensor, ...]): Ray directions in the local camera frame. (B, 3, H, W)
        ray_depths (Tuple[torch.Tensor, ...]): Depth along the ray. (B, 1, H, W)
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

    def forward(
        self, views: List[torch.Tensor]
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        num_views = len(views)
        x = torch.cat(views, dim=0)  # (N*B, 3, H, W)
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
        ray_directions = torch.chunk(ray_directions, num_views, dim=0)
        ray_depths = torch.chunk(ray_depths, num_views, dim=0)
        return ray_directions, ray_depths
