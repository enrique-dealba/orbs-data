import json
import os
from typing import Dict, List, Tuple

import dvc.api
import numpy as np
from sklearn.model_selection import train_test_split


def load_preprocessed_data() -> Dict[str, np.ndarray]:
    """Loads preprocessed .npy files from data/preprocessed dir."""
    data = {}
    preprocessed_dir = "data/preprocessed"
    for file in os.listdir(preprocessed_dir):
        if file.endswith(".npy"):
            with dvc.api.open(os.path.join(preprocessed_dir, file), mode="rb") as f:
                data[file] = np.load(f)
    return data


def combine_data(data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Combine all data arrays into a single numpy array.
    Assumes all arrays have the same shape except for the first dimension.
    """
    return np.concatenate(list(data.values()), axis=0)


def split_data(
    data: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the combined data into training and validation sets.
    Maintains the temporal structure by splitting along the first axis.
    """
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    return data[train_indices], data[val_indices]


def save_data(
    train_data: np.ndarray, val_data: np.ndarray, output_dir: str = "data/split"
):
    """Saves train and val sets to specified output dir."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train_data.npy"), train_data)
    np.save(os.path.join(output_dir, "val_data.npy"), val_data)


def save_split_info(
    train_indices: List[int], val_indices: List[int], output_dir: str = "data/split"
):
    """Saves info about train/val split for reproducibility."""
    split_info = {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
    }
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f)


def main():
    # Load all preprocessed data
    data = load_preprocessed_data()

    # Combine all data
    combined_data = combine_data(data)

    # Split data into train and validation sets
    train_data, val_data = split_data(combined_data)

    # Save the split data
    save_data(train_data, val_data)

    # Save split information
    n_samples = combined_data.shape[0]
    indices = np.arange(n_samples)
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    save_split_info(train_indices, val_indices)

    print(
        f"Data split complete. Shape of train data: {train_data.shape}, Shape of validation data: {val_data.shape}"
    )


if __name__ == "__main__":
    main()
