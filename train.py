from typing import Tuple

import dvc.api
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from utils import save_model, visualize_reconstruction
from vae_cnn import VAECNN


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def load_split_data() -> Tuple[torch.Tensor, torch.Tensor]:
    with dvc.api.open("data/split/train_data.npy", mode="rb") as f:
        train_data = np.load(f)
    with dvc.api.open("data/split/val_data.npy", mode="rb") as f:
        val_data = np.load(f)

    # Convert to PyTorch tensors and permute dimensions
    train_data = torch.from_numpy(train_data).float().permute(0, 3, 1, 2)
    val_data = torch.from_numpy(val_data).float().permute(0, 3, 1, 2)

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
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon_x, mu, logvar = model(x)
                loss = vae_loss(recon_x, x, mu, logvar)
                val_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    return model, train_loss, val_loss


def main():
    # Hyperparameters
    latent_dim = 32
    batch_size = 2
    learning_rate = 3e-4
    num_epochs = 10
    encoder_channels = [4, 8, 16, 32]  # Customizable

    wandb.init(
        project="vae-convlstm-noise",
        config={
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "encoder_channels": encoder_channels,
        },
    )

    train_data, val_data = load_split_data()

    # Create datasets without assuming labels
    train_dataset = torch.utils.data.TensorDataset(train_data)
    val_dataset = torch.utils.data.TensorDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_shape = train_data.shape[1:]  # Now will be (3, 572, 217)
    model = VAECNN(input_shape, latent_dim, encoder_channels)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, final_train_loss, final_val_loss = train_model(
        model, train_loader, val_loader, optimizer, device, num_epochs
    )

    final_metrics = {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    save_model(model, wandb.config, final_metrics)

    # Visualize reconstructions for a few samples
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 4:  # Visualize first 4 samples
                break
            x = batch[0].to(device)  # (batch_size, channels, height, width)
            filename = f"reconstruction_{i}.png"
            visualize_reconstruction(
                model,
                x,
                save_dir="reconstructions",
                filename=filename,
                log_to_wandb=True,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
