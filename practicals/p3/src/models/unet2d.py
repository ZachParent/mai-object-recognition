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
        activation="relu",
        batch_norm=False,
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


class UNetLeft(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stack_num=2,
        activation="relu",
        pool=True,
        batch_norm=False,
    ):
        super().__init__()
        self.pool = pool

        # Downsampling
        if pool:
            self.down = nn.MaxPool2d(2)
        else:
            self.down = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

        # Convolution stack
        self.conv = ConvBlock(
            in_channels, out_channels, kernel_size, stack_num, activation, batch_norm
        )

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UNetRight(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stack_num=2,
        activation="relu",
        unpool=True,
        batch_norm=False,
    ):
        super().__init__()

        # Upsampling
        if unpool:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)

        # Convolution before concatenation
        self.conv_before = ConvBlock(
            in_channels, out_channels, kernel_size, 1, activation, batch_norm
        )

        # Convolution after concatenation
        self.conv_after = ConvBlock(
            out_channels * 2,
            out_channels,
            kernel_size,
            stack_num,
            activation,
            batch_norm,
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_before(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_after(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        input_size,
        filter_num,
        n_labels,
        stack_num_down=2,
        stack_num_up=2,
        activation="ReLU",
        output_activation="Softmax",
        batch_norm=False,
        pool=True,
        unpool=True,
    ):
        super().__init__()

        # Extract in_channels from input_size
        if isinstance(input_size, tuple):
            in_channels = input_size[2] if len(input_size) == 3 else input_size[0]
        else:
            in_channels = input_size

        self.depth = len(filter_num)

        # Initial convolution
        self.init_conv = ConvBlock(
            in_channels,
            filter_num[0],
            stack_num=stack_num_down,
            activation=activation.lower(),
            batch_norm=batch_norm,
        )

        # Downsampling path
        self.down_path = nn.ModuleList(
            [
                UNetLeft(
                    filter_num[i],
                    filter_num[i + 1],
                    stack_num=stack_num_down,
                    activation=activation.lower(),
                    pool=pool,
                    batch_norm=batch_norm,
                )
                for i in range(self.depth - 1)
            ]
        )

        # Upsampling path
        self.up_path = nn.ModuleList(
            [
                UNetRight(
                    filter_num[-i - 1],
                    filter_num[-i - 2],
                    stack_num=stack_num_up,
                    activation=activation.lower(),
                    unpool=unpool,
                    batch_norm=batch_norm,
                )
                for i in range(self.depth - 1)
            ]
        )

        # Output layer
        self.output = nn.Conv2d(filter_num[0], n_labels, 1)
        self.output_activation = output_activation

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        skip_connections = [x]

        # Downsampling path
        for down in self.down_path:
            x = down(x)
            skip_connections.append(x)

        # Remove last skip connection as it's not needed
        skip_connections.pop()

        # Upsampling path
        for up, skip in zip(self.up_path, reversed(skip_connections)):
            x = up(x, skip)

        # Output layer
        x = self.output(x)

        # Apply output activation
        if self.output_activation == "Softmax":
            x = F.softmax(x, dim=1)
        elif self.output_activation == "Sigmoid":
            x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = UNet2D(input_size=(256, 256, 3), filter_num=[64, 128, 256, 512], n_labels=1)

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
