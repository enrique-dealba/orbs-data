from typing import Tuple

import dvc.api
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, TensorDataset

from small_vae import VAEConvLSTM
from utils import save_model


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD


def load_split_data() -> Tuple[np.ndarray, np.ndarray]:
    with dvc.api.open("data/split/train_data.npy", mode="rb") as f:
        train_data = np.load(f)
    with dvc.api.open("data/split/val_data.npy", mode="rb") as f:
        val_data = np.load(f)
    return train_data, val_data


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
) -> Tuple[nn.Module, float, float]:
    model.to(device)
    final_train_loss = 0.0
    final_val_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch[0].to(device)  # Assuming the loader returns a tuple
            recon_x, mean, logvar = model(x)
            loss = vae_loss(recon_x, x, mean, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)  # Assuming the loader returns a tuple
                recon_x, mean, logvar = model(x)
                loss = vae_loss(recon_x, x, mean, logvar)
                val_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        final_train_loss = train_loss  # Update final losses
        final_val_loss = val_loss

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    return model, final_train_loss, final_val_loss


def main():
    # Hyperparameters
    latent_dim = 16
    batch_size = 2
    learning_rate = 3e-4
    num_epochs = 5

    # Initialize wandb
    wandb.init(
        project="vae-convlstm-noise",
        config={
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        },
    )

    # Load and preprocess data
    train_data, val_data = load_split_data()
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_shape = train_data.shape[1:]  # (frames, height, width, channels)
    model = VAEConvLSTM(input_shape, latent_dim)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, final_train_loss, final_val_loss = train_model(
        model, train_loader, val_loader, optimizer, device, num_epochs
    )

    # Save model with unique identifier
    final_metrics = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    save_model(model, wandb.config, final_metrics)

    wandb.finish()


if __name__ == "__main__":
    main()
