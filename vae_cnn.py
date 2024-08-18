import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, channel_dim, height, width):
        super(UnFlatten, self).__init__()
        self.channel_dim = channel_dim
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel_dim, self.height, self.width)


class CustomPadding(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CustomPadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        h, w = x.shape[2:]
        pad_h = max(
            (math.ceil(h / self.stride) * self.stride - h + self.kernel_size - 1) // 2,
            0,
        )
        pad_w = max(
            (math.ceil(w / self.stride) * self.stride - w + self.kernel_size - 1) // 2,
            0,
        )
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h))


class VAECNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        encoder_channels: List[int] = [32, 64, 128, 256],
        kernel_size: int = 4,
        stride: int = 2,
    ):
        super(VAECNN, self).__init__()
        self.input_shape = input_shape  # Expected to be (channels, height, width)
        self.latent_dim = latent_dim
        self.encoder_channels = [input_shape[0]] + encoder_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Calculate the dimension after flattening
        self.encoder_output_height = math.ceil(
            input_shape[1] / (stride ** len(encoder_channels))
        )
        self.encoder_output_width = math.ceil(
            input_shape[2] / (stride ** len(encoder_channels))
        )
        self.flat_dim = (
            encoder_channels[-1]
            * self.encoder_output_height
            * self.encoder_output_width
        )

        # print(f"Expected flat_dim: {self.flat_dim}")

        # Build encoder
        encoder_layers = []
        for i in range(len(self.encoder_channels) - 1):
            encoder_layers.extend(
                [
                    CustomPadding(kernel_size, stride),
                    nn.Conv2d(
                        self.encoder_channels[i],
                        self.encoder_channels[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                    ),
                    nn.ReLU(),
                ]
            )
        encoder_layers.append(Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Build decoder
        decoder_channels = encoder_channels[::-1] + [input_shape[0]]
        decoder_layers = [
            UnFlatten(
                encoder_channels[-1],
                self.encoder_output_height,
                self.encoder_output_width,
            )
        ]
        for i in range(len(decoder_channels) - 1):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        decoder_channels[i],
                        decoder_channels[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1,
                    ),
                    nn.ReLU() if i < len(decoder_channels) - 2 else nn.Sigmoid(),
                ]
            )
        decoder_layers.append(nn.AdaptiveAvgPool2d((input_shape[1], input_shape[2])))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        # print(f"Encoder output shape: {h.shape}")
        # print(
        #     f"Expected encoder output shape: {torch.Size([x.size(0), self.flat_dim])}"
        # )
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # print(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}")
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        # print(f"Decoder input shape: {h.shape}")
        # print(f"Expected decoder input shape: {torch.Size([z.size(0), self.flat_dim])}")
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def custom_mse_loss(output, target):
    return F.mse_loss(output, target, reduction="mean")


def print_model_summary(model):
    print("\nModel Summary:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")
