from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class DPT(nn.Module):
    """
    DPT for dense prediction
    """

    def __init__(
        self,
        in_channels: int,
        output_dim: int = 4,
        patch_size: int = 14,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = output_dim

        # Channel projection: in_channels -> out_channels[i]
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        # Spatial resize: [x4, x2, x1, /2]
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Fusion blocks
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Heads
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                head_features_2, self.out_dim, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(
        self,
        features: List[Tuple[torch.Tensor, torch.Tensor]],
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for i, (patch, _cls) in enumerate(features):
            B, _, C = patch.shape
            x = patch.permute(0, 2, 1).contiguous().reshape(B, C, patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        out = self._fuse(out)

        x = self.scratch.output_conv(out)
        x = nn.functional.interpolate(x, (H, W), mode="bilinear", align_corners=True)
        x = self.scratch.output_conv2(x)

        ray_directions, ray_depths = torch.split(x, [3, 1], dim=1)
        ray_directions = nn.functional.normalize(ray_directions, dim=1)
        ray_depths = torch.exp(ray_depths)
        return ray_directions, ray_depths

    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        l1 = self.scratch.layer1_rn(feats[0])
        l2 = self.scratch.layer2_rn(feats[1])
        l3 = self.scratch.layer3_rn(feats[2])
        l4 = self.scratch.layer4_rn(feats[3])

        path = self.scratch.refinenet4(l4, size=l3.shape[2:])
        path = self.scratch.refinenet3(path, l3, size=l2.shape[2:])
        path = self.scratch.refinenet2(path, l2, size=l1.shape[2:])
        path = self.scratch.refinenet1(path, l1)
        return path


def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(features, has_residual=has_residual)


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(
        self, features: int, size: Tuple[int, int] = None, has_residual: bool = True
    ) -> None:
        super().__init__()
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)

        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        output = xs[0]

        if self.has_residual and len(xs) == 2 and self.resConfUnit1 is not None:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=True
        )

        output = self.out_conv(output)
        return output
