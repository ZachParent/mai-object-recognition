from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet2d import ConvBlock, UNetLeft, UNetRight


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()  # drop_path if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = (
            self.patch_embed(x).flatten(2).transpose(1, 2)
        )  # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 1:]  # Remove CLS token

    def forward(self, x):
        return self.forward_features(x)


def trunc_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class TransUNet(nn.Module):
    def __init__(
        self,
        input_size,
        filter_num: List[int],
        n_labels: int,
        stack_num_down: int = 2,
        stack_num_up: int = 2,
        activation: str = "ReLU",
        output_activation: str = "Softmax",
        batch_norm: bool = False,
        pool: bool = True,
        unpool: bool = True,
        # Vision Transformer parameters
        embed_dim: int = 768,  # Corresponds to proj_dim in keras-unet-collection
        vit_patch_size: int = 16,
        vit_num_heads: int = 12,
        vit_num_layers: int = 12,
        vit_mlp_ratio: float = 4.0,
        vit_qkv_bias: bool = True,
        vit_qk_scale: float | None = None,
        vit_drop_rate: float = 0.0,
        vit_attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        # Extract in_channels from input_size
        in_channels = input_size[2]
        self.vit_input_size = input_size[
            0:2
        ]  # Tuple (H, W) of the input image for ViT dimension calculation

        self.depth = len(filter_num)
        self.vit_patch_size = vit_patch_size

        # Initial convolution for CNN
        self.init_conv = ConvBlock(
            in_channels,
            filter_num[0],
            stack_num=stack_num_down,
            activation=activation.lower(),
            batch_norm=batch_norm,
        )

        # Downsampling path (CNN encoder)
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

        # Vision Transformer as bottleneck
        # Calculate the H, W of the feature map fed to ViT
        self.cnn_feat_dim = input_size[0] // (2 ** (self.depth - 1))

        self.vit = VisionTransformer(
            img_size=self.cnn_feat_dim,
            patch_size=vit_patch_size,
            in_chans=filter_num[-1],  # Channels from CNN encoder output
            embed_dim=embed_dim,  # ViT's own embedding dimension
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            qkv_bias=vit_qkv_bias,
            qk_scale=vit_qk_scale,
            drop_rate=vit_drop_rate,
            attn_drop_rate=vit_attn_drop_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # Projection from ViT output back to CNN feature map shape for decoder
        projection_layers = []
        # 1. 1x1 Conv: embed_dim -> filter_num[-1]
        projection_layers.append(nn.Conv2d(embed_dim, filter_num[-1], kernel_size=1))
        if batch_norm:
            projection_layers.append(nn.BatchNorm2d(filter_num[-1]))
        projection_layers.append(
            nn.ReLU() if activation.lower() == "relu" else nn.GELU()
        )

        # 2. 3x3 Conv: filter_num[-1] -> filter_num[-1] (with padding to maintain size)
        projection_layers.append(
            nn.Conv2d(filter_num[-1], filter_num[-1], kernel_size=3, padding=1)
        )
        if batch_norm:
            projection_layers.append(nn.BatchNorm2d(filter_num[-1]))
        projection_layers.append(
            nn.ReLU() if activation.lower() == "relu" else nn.GELU()
        )

        self.vit_to_cnn_projection = nn.Sequential(*projection_layers)

        # Upsampling path (CNN decoder)
        self.up_path = nn.ModuleList()
        for i in range(self.depth - 1):
            self.up_path.append(
                UNetRight(
                    filter_num[-i - 1],
                    filter_num[-i - 2],
                    stack_num=stack_num_up,
                    activation=activation.lower(),
                    unpool=unpool,
                    batch_norm=batch_norm,
                )
            )

        # Output layer
        self.output_conv = nn.Conv2d(filter_num[0], n_labels, 1)
        self.output_activation_fn = None
        self.output_activation = output_activation

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        skip_connections = [x]

        # Downsampling path (CNN Encoder)
        for i, down_layer in enumerate(self.down_path):
            x = down_layer(x)
            skip_connections.append(x)

        # Remove last skip connection as it's not needed
        skip_connections.pop()

        # Bottleneck (Vision Transformer)
        x_vit_embedded = self.vit.forward_features(x)  # (B, num_patches, embed_dim)

        # Reshape ViT output to be image-like for the decoder
        B, N, E = x_vit_embedded.shape

        x_vit_reshaped = x_vit_embedded.transpose(1, 2).reshape(
            B, E, self.cnn_feat_dim, self.cnn_feat_dim
        )  # (B, embed_dim, H_feat, W_feat)

        # Project ViT output to match decoder's expected channels
        x = self.vit_to_cnn_projection(x_vit_reshaped)

        # Upsampling path (CNN Decoder)
        for up, skip in zip(self.up_path, reversed(skip_connections)):
            x = up(x, skip)

        # Output layer
        x = self.output_conv(x)

        # Apply output activation
        if self.output_activation == "Softmax":
            x = F.softmax(x, dim=1)
        elif self.output_activation == "Sigmoid":
            x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    # Example configuration similar to TransUNet paper for R50-ViT-B_16
    # Input size (224, 224, 3)
    # Filter numbers for CNN encoder part (e.g., ResNet50 like)
    # For simplicity, we use the U-Net filter numbers, but in TransUNet, this comes from a pre-trained CNN (like ResNet)
    # The last filter_num should match the input channels to ViT if not using a separate projection.
    # Here, filter_num[-1] is the number of channels input to ViT.
    # For TransUNet, the CNN encoder part is often a pre-trained model like ResNet-50.
    # The `filter_num` would correspond to the feature channels at different stages of that ResNet.
    # For example, if ResNet50 is used, after its 4 stages, the channels might be [64, 256, 512, 1024, 2048] (roughly)
    # And the ViT would operate on the output of the last stage.
    # The `depth` of the U-Net here corresponds to how many downsampling steps are in the CNN part.
    # If depth is 5, there are 4 downsampling UNetLeft blocks.
    # The ViT input size would be input_img_size / (2^4) = 224 / 16 = 14.
    # So ViT patch size should be a divisor of 14, e.g., 1 or 2.
    # The original TransUNet uses ViT-B/16, meaning patch size 16 on the *original* image.
    # If the CNN encoder reduces resolution by 16x (e.g. 224 -> 14), then the ViT operates on this 14x14 feature map.
    # The "patches" for the ViT are then taken from this 14x14 feature map.
    # If vit_patch_size=1 for the ViT operating on the 14x14 map, it means each "pixel" of the 14x14 map is a token.
    # Let's adjust parameters for a TransUNet-like setup:
    # Input image 224x224x3
    # CNN encoder downsamples 4 times (depth=5 for filter_num)
    # Feature map to ViT: 224 / (2^4) = 224 / 16 = 14x14
    # ViT patch size on this 14x14 map. Let's use 1, so each 1x1 pixel in the 14x14 map is a patch.
    # Number of patches = 14*14 = 196.
    # ViT embed_dim = 768 (standard for ViT-Base)

    image_size = 224
    vit_ps = 1  # ViT patch size on the feature map from CNN encoder

    model = TransUNet(
        input_size=(image_size, image_size, 3),  # H, W, C
        filter_num=[64, 128, 256, 512, 768],  # Last one is input to ViT if no backbone
        n_labels=1,  # Example: binary segmentation
        stack_num_down=2,
        stack_num_up=2,  # TransUNet uses 2 generally
        activation="ReLU",  # ReLU in CNN, GELU in ViT (handled by ViT block)
        output_activation="Sigmoid",
        batch_norm=True,
        pool=True,
        unpool=True,  # TransUNet uses bilinear upsampling + convs
        # TransUNet specific
        embed_dim=768,  # ViT embedding dimension
        vit_patch_size=vit_ps,  # Patch size for ViT on the bottleneck features
        vit_num_heads=12,  # Standard for ViT-Base
        vit_num_layers=12,  # Standard for ViT-Base (e.g. ViT-B/16 has 12 layers)
        vit_mlp_ratio=4.0,
        vit_qkv_bias=True,  # In timm, ViT uses qkv_bias=True
        vit_drop_rate=0.0,
        vit_attn_drop_rate=0.0,
    )

    # The summary might be very large.
    # Let's try with a smaller input for summary if needed, or just print model.
    # The input to summary should be (batch_size, C, H, W)
    print("Model instantiated. Attempting summary...")
    try:
        summary(
            model, (1, 3, image_size, image_size), device="cpu"
        )  # Add device="cpu" if no CUDA
    except Exception as e:
        print(f"Could not generate summary: {e}")
        print(model)

    # Test with a dummy input
    print("\\nTesting with a dummy input:")
    dummy_video = torch.randn(1, 3, image_size, image_size)  # B, C, H, W
    print(f"Input video shape: {dummy_video.shape}")

    # No einops packing needed if we pass single images or batches of images directly
    # packed_video, packing_config = einops.pack([video], "* channel height width")
    # print(f"Packed video shape: {packed_video.shape}")

    # Pass packed video to model
    try:
        with torch.no_grad():  # Ensure no gradients are computed during test forward pass
            preds = model(dummy_video)
        print(f"Preds shape: {preds.shape}")  # Expected: (B, n_labels, H, W)
    except Exception as e:
        print(f"Error during model forward pass: {e}")

    # unpacked_preds = einops.unpack(preds, packing_config, "* channel height width")[0]
    # print(f"Unpacked preds shape: {unpacked_preds.shape}")

    # Further check: TransUNet often uses a pre-trained backbone (e.g. ResNet50) as the CNN encoder.
    # This implementation uses a generic U-Net style CNN encoder.
    # To fully replicate TransUNet with a backbone, the UNetLeft/init_conv part would be replaced by the backbone,
    # and skip connections would be extracted from intermediate layers of that backbone.
    # The ViT would then process the output of the backbone's final feature stage.
    # The decoder would then upscale this, combining with skip connections from the backbone.
    # This current version is a U-Net with a ViT at the bottleneck, which is the core idea of TransUNet
    # without the specific pre-trained backbone aspect.
