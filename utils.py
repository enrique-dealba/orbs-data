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


import matplotlib.pyplot as plt

def visualize_reconstruction(model, input_tensor):
    with torch.no_grad():
        output, _, _ = model(input_tensor)
    
    # Convert tensors to numpy arrays and reshape
    input_img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(input_img)
    ax1.set_title('Original')
    ax2.imshow(output_img)
    ax2.set_title('Reconstruction')
    plt.show()

# Use in your test_model function
# visualize_reconstruction(model, random_input)
