from typing import Tuple

import torch
import torch.nn as nn

from conv_lstm import ConvLSTMCell


class VAEConvLSTM(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int, int], latent_dim: int):
        super(VAEConvLSTM, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Play around with these
        hidden_dim1 = 8
        hidden_dim2 = 16

        # Encoder
        self.encoder = nn.Sequential(
            ConvLSTMCell(
                input_dim=input_shape[-1],
                hidden_dim=hidden_dim1,
                kernel_size=3,
                bias=True,
            ),
            nn.MaxPool2d(
                4, 4
            ),  # Increased pooling to reduce spatial dimensions more quickly
            ConvLSTMCell(
                input_dim=hidden_dim1, hidden_dim=hidden_dim2, kernel_size=3, bias=True
            ),
            nn.MaxPool2d(4, 4),  # Increased pooling again
            nn.Flatten(),
            nn.Linear(
                hidden_dim2 * (input_shape[1] // 16) * (input_shape[2] // 16),
                latent_dim * 2,
            ),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim2 * (input_shape[1] // 16) * (input_shape[2] // 16),
            ),
            nn.Unflatten(1, (hidden_dim2, input_shape[1] // 16, input_shape[2] // 16)),
            nn.ConvTranspose2d(
                hidden_dim2,
                hidden_dim1,
                kernel_size=3,
                stride=4,
                padding=1,
                output_padding=3,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim1,
                input_shape[-1],
                kernel_size=3,
                stride=4,
                padding=1,
                output_padding=3,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(x)
        mean, logvar = torch.chunk(result, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
