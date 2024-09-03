import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb


def save_model(model, config, metrics):
    # Creates unique id for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_channels = str('-'.join(map(str, config.encoder_channels)))
    run_name = f"vae_cnn_{timestamp}_ld{config.latent_dim}_ec{encoder_channels}_bs{config.batch_size}_lr{config.learning_rate:.1e}"

    # Create a directory for saving models if doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the model
    model_path = os.path.join("models", f"{run_name}.pth")

    # torch.save(
    #     {"model_state_dict": model.state_dict(), "config": config, "metrics": metrics},
    #     model_path,
    # )

    print(f"Saving model: {run_name}.pth to models/.")

    config_dict = dict(config)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config': config_dict,
            'metrics': metrics
        },
        f'models/{run_name}.pth'
    )

    # Log the model in wandb
    print(f"Saving model: {model_path} to wandb.")
    wandb.save(model_path)

    return model_path


def visualize_reconstruction(
    model, input_tensor, save_dir="reconstructions", filename=None, log_to_wandb=True
):
    model.eval()
    with torch.no_grad():
        output, _, _ = model(input_tensor)

    # Ensure we're working with the first item in the batch
    input_img = input_tensor[0].permute(1, 2, 0).cpu().numpy()
    output_img = output[0].permute(1, 2, 0).cpu().numpy()

    # Clip values to [0, 1] range
    input_img = np.clip(input_img, 0, 1)
    output_img = np.clip(output_img, 0, 1)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_img)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(output_img)
    ax2.set_title("Reconstruction")
    ax2.axis("off")
    plt.tight_layout()

    # Save locally if directory is provided
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            if filename is None:
                filename = f"reconstruction_{wandb.run.id}_{wandb.run.step}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    # Log to wandb
    if log_to_wandb:
        try:
            wandb_image = wandb.Image(fig, caption="Original vs Reconstruction")
            wandb.log({"reconstruction": wandb_image})
            print("Image logged to wandb")
        except Exception as e:
            print(f"Error logging to wandb: {e}")

    plt.close(fig)

    return input_img, output_img
