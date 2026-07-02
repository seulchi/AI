from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


from .dpt import DPT


class ModelNet(nn.Module):
    """
    Architecture:
    - Backbone: DinoV2 feature extractor with alternating attention
    - Head: DPT for factored ray prediction
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        backbone: nn.Module,
        intermediate_layer_idx: Sequence[int] = (11, 15, 19, 23),
        dpt_features: int = 256,
        dpt_out_channels: Sequence[int] = (256, 512, 1024, 1024),
        dpt_output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.intermediate_layer_idx = list(intermediate_layer_idx)
        self.dpt = DPT(
            in_channels=backbone.embed_dim * 2,
            output_dim=dpt_output_dim,
            patch_size=self.PATCH_SIZE,
            features=dpt_features,
            out_channels=dpt_out_channels,
        )

    def forward(self, views: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            views (List[torch.Tensor]): List of input views' images. (B, 3, H, W)

        Returns:
            ray_directions (torch.Tensor): Ray directions in the local camera frame. (B, S, 3, H, W)
            ray_depths (torch.Tensor): Depth along the ray. (B, S, 1, H, W)

        """

        num_of_views = len(views)
        B, _, H, W = views[0].shape
        x = torch.stack(views, dim=1)

        features = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layer_idx,
            norm=True,
        )
        # features: [(feature (B, S, P, 2C), camera_token (B, S, 2C)), ...]

        ray_directions, ray_depths = self.dpt(features, H, W)

        ray_directions = ray_directions.unflatten(0, (B, num_of_views))
        ray_depths = ray_depths.unflatten(0, (B, num_of_views))

        return ray_directions, ray_depths
