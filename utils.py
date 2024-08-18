import os
from datetime import datetime

import torch
import wandb


def save_model(model, config, metrics):
    # Create a unique identifier for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vae_convlstm_{timestamp}_ld{config.latent_dim}_bs{config.batch_size}_lr{config.learning_rate:.1e}"

    # Create a directory for saving models if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the model
    model_path = os.path.join("models", f"{run_name}.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "metrics": metrics},
        model_path,
    )

    # Log the model in wandb
    wandb.save(model_path)

    print(f"Model saved as {model_path}")
    return model_path
