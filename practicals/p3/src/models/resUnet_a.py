# Adapted from https://github.com/Akhilesh64/ResUnet-a/blob/main/model.py
from typing import Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stack_num=2,
        activation="GELU",
        batch_norm=True,
    ):
        super().__init__()
        self.stack_num = stack_num
        self.activation = activation
        self.batch_norm = batch_norm

        # Create stack of conv layers
        layers = []
        for i in range(stack_num):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU() if activation == "relu" else nn.GELU())

        self.conv_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_stack(x)


def create_normalization_layer(num_features, layer_norm="batch"):
    """Helper function to create appropriate normalization layer"""
    if layer_norm == "batch":
        return nn.BatchNorm2d(num_features)
    else:
        return nn.InstanceNorm2d(num_features, affine=True)


class ResUnetA(nn.Module):
    """PyTorch implementation of ResUnet-a architecture adapted for depth estimation"""

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        n_labels: int = 1,  # Default to 1 for depth map
        layer_norm: str = "batch",
        output_activation: str = "Sigmoid",  # Default to None for depth map
    ):
        super().__init__()

        # Extract input channels from input_size
        self.height, self.width = input_size[0], input_size[1]
        self.channels = input_size[2]

        self.num_classes = n_labels
        self.layer_norm = layer_norm
        self.output_activation = output_activation

        # Layer 1: Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.channels, 32, 1, padding=0),
            create_normalization_layer(32, self.layer_norm),
            nn.ReLU(),
        )

        # Layer 2: First residual block
        self.res_block1 = self._residual_block(32, 32, [1, 3, 15, 31])

        # Layer 3: Downsampling
        self.down1 = nn.Conv2d(32, 64, 1, stride=2, padding=0)

        # Layer 4: Second residual block
        self.res_block2 = self._residual_block(64, 64, [1, 3, 15, 31])

        # Layer 5: Downsampling
        self.down2 = nn.Conv2d(64, 128, 1, stride=2, padding=0)

        # Layer 6: Third residual block
        self.res_block3 = self._residual_block(128, 128, [1, 3, 15])

        # Layer 7: Downsampling
        self.down3 = nn.Conv2d(128, 256, 1, stride=2, padding=0)

        # Layer 8: Fourth residual block
        self.res_block4 = self._residual_block(256, 256, [1, 3, 15])

        # Layer 9: Downsampling
        self.down4 = nn.Conv2d(256, 512, 1, stride=2, padding=0)

        # Layer 10: Fifth residual block
        self.res_block5 = self._residual_block(512, 512, [1])

        # Layer 11: Downsampling
        self.down5 = nn.Conv2d(512, 1024, 1, stride=2, padding=0)

        # Layer 12: Sixth residual block
        self.res_block6 = self._residual_block(1024, 1024, [1])

        # Additional layers for deeper model variant
        self.down6 = nn.Conv2d(1024, 2048, 1, stride=2, padding=0)
        self.res_block7 = self._residual_block(2048, 2048, [1])
        self.psp_pooling_deep = self._psp_pooling(2048, 2048)

        # Upsampling and combining paths
        self.combine1 = self._combine(2048, 1024, 1024)
        self.combine2 = self._combine(1024, 512, 512)
        self.up_res_block1 = self._residual_block(512, 512, [1])
        self.combine3 = self._combine(512, 256, 256)
        self.up_res_block2 = self._residual_block(256, 256, [1])
        self.combine4 = self._combine(256, 128, 128)
        self.up_res_block3 = self._residual_block(128, 128, [1])
        self.combine5 = self._combine(128, 64, 64)
        self.up_res_block4 = self._residual_block(64, 64, [1])
        self.combine6 = self._combine(64, 32, 32)
        self.up_res_block5 = self._residual_block(32, 32, [1])

        # Final PSP Pooling
        self.final_psp = self._psp_pooling(64, 32)  # 64 = 32 + 32 (x + x1)

        # Depth output branch - simplified from the original three branches
        self.depth_conv1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 32, 3),  # Input from concatenated features
            create_normalization_layer(32, self.layer_norm),
            nn.ReLU(),
        )
        self.depth_conv2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(32, 32, 3),
            create_normalization_layer(32, self.layer_norm),
            nn.ReLU(),
        )
        # Final output layer for depth map
        self.depth_output = nn.Conv2d(32, self.num_classes, 1)

    def _residual_block(self, in_channels, out_channels, dilation_rates):
        """Implement the residual block with dilated convolutions"""
        return ResidualBlock(in_channels, out_channels, dilation_rates, self.layer_norm)

    def _psp_pooling(self, in_channels, out_channels):
        """Implement Pyramid Scene Parsing (PSP) pooling"""
        return PSPPooling(in_channels, out_channels, self.layer_norm)

    def _combine(self, in_channels, skip_channels, out_channels):
        """Combine upsampled features with skip connections"""
        return CombineLayer(in_channels, skip_channels, out_channels, self.layer_norm)

    def forward(self, x):
        # Encoder path with skip connections
        x1 = self.initial_conv(x)  # Layer 1
        x2 = self.res_block1(x1)  # Layer 2
        x3 = self.down1(x2)  # Layer 3
        x4 = self.res_block2(x3)  # Layer 4
        x5 = self.down2(x4)  # Layer 5
        x6 = self.res_block3(x5)  # Layer 6
        x7 = self.down3(x6)  # Layer 7
        x8 = self.res_block4(x7)  # Layer 8
        x9 = self.down4(x8)  # Layer 9
        x10 = self.res_block5(x9)  # Layer 10
        x11 = self.down5(x10)  # Layer 11
        x12 = self.res_block6(x11)  # Layer 12

        # ResUnet-a d7 model
        x = self.down6(x12)
        x = self.res_block7(x)
        x = self.psp_pooling_deep(x)
        x = self.combine1(x, x12)

        # Decoder path
        x = self.combine2(x, x10)
        x = self.up_res_block1(x)
        x = self.combine3(x, x8)
        x = self.up_res_block2(x)
        x = self.combine4(x, x6)
        x = self.up_res_block3(x)
        x = self.combine5(x, x4)
        x = self.up_res_block4(x)
        x = self.combine6(x, x2)
        x = self.up_res_block5(x)

        # Concatenate with initial features
        x_cat = torch.cat([x, x1], dim=1)

        # Process through depth estimation branch
        depth = self.depth_conv1(x_cat)
        depth = self.depth_conv2(depth)
        depth = self.depth_output(depth)

        # Apply output activation if specified
        if self.output_activation == "Softmax":
            depth = F.softmax(depth, dim=1)
        elif self.output_activation == "Sigmoid":
            depth = torch.sigmoid(depth)

        return depth


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions"""

    def __init__(self, in_channels, out_channels, dilation_rates, layer_norm):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.layer_norm = layer_norm

        # Create a branch for each dilation rate
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            padding = rate  # For dilated convs, padding = dilation rate to maintain spatial dims
            branch = nn.Sequential(
                create_normalization_layer(in_channels, layer_norm),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, dilation=rate, padding=padding),
                create_normalization_layer(out_channels, layer_norm),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels, out_channels, 3, dilation=rate, padding=padding
                ),
            )
            self.branches.append(branch)

    def forward(self, x):
        outputs = [x]  # Include input in the outputs to be added
        for branch in self.branches:
            outputs.append(branch(x))
        return torch.sum(torch.stack(outputs), dim=0)  # Sum all branches


class PSPPooling(nn.Module):
    """Pyramid Scene Parsing Pooling module"""

    def __init__(self, in_channels, out_channels, layer_norm):
        super().__init__()
        self.in_channels = in_channels

        # Create pooling paths for different scales
        self.paths = nn.ModuleList()
        for pool_size in [1, 2, 4]:
            path = nn.Sequential(
                nn.MaxPool2d(pool_size, stride=pool_size),
                nn.Upsample(scale_factor=pool_size, mode="nearest"),
                nn.Conv2d(in_channels, out_channels // 4, 1, padding=0),
                create_normalization_layer(out_channels // 4, layer_norm),
            )
            self.paths.append(path)

        # Projection convolution
        # The input to projection is the original input (in_channels) plus the outputs from the 3 paths
        proj_in_channels = in_channels + (out_channels // 4) * 3
        self.projection = nn.Sequential(
            nn.Conv2d(proj_in_channels, out_channels, 1, padding=0),
            create_normalization_layer(out_channels, layer_norm),
        )

    def forward(self, x):
        # Apply pooling at different scales
        pool_results = [x]
        for path in self.paths:
            pool_results.append(path(x))

        # Handle possible size mismatch due to padding issues
        sizes = [p.size()[2:] for p in pool_results]
        min_h = min(s[0] for s in sizes)
        min_w = min(s[1] for s in sizes)

        # Resize to smallest dimension if needed
        for i in range(len(pool_results)):
            if pool_results[i].size(2) > min_h or pool_results[i].size(3) > min_w:
                pool_results[i] = F.interpolate(
                    pool_results[i],
                    size=(min_h, min_w),
                    mode="bilinear",
                    align_corners=True,
                )

        # Concatenate and project
        x = torch.cat(pool_results, dim=1)
        return self.projection(x)


class CombineLayer(nn.Module):
    """Layer for combining upsampled features with skip connections"""

    def __init__(self, in_channels, skip_channels, out_channels, layer_norm):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, padding=0),
            create_normalization_layer(out_channels, layer_norm),
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle size mismatch if upsampling doesn't exactly match skip connection
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(
                x, size=skip.size()[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.projection(x)


if __name__ == "__main__":
    from torchinfo import summary

    model = ResUnetA(
        input_size=(256, 256, 3),
    )

    summary(model, (1, 3, 256, 256))

    # 2 videos, 2 frames, 3 channels, 256x256
    video = torch.randn(2, 2, 3, 256, 256)
    print(f"Video shape: {video.shape}")
    packed_video, packing_config = einops.pack([video], "* channel height width")
    print(f"Packed video shape: {packed_video.shape}")
    # pass packed video to model
    preds = model(packed_video)
    print(f"Packed preds shape: {preds.shape}")
    unpacked_preds = einops.unpack(preds, packing_config, "* channel height width")[0]
    print(f"Unpacked preds shape: {unpacked_preds.shape}")
