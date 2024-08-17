import json
import os
from typing import Dict, Tuple

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


def load_preprocessed_data() -> Dict[str, NDArray]:
    preprocessed_files = [
        "control1_stack_normalized.npy",
        "control2_stack_normalized.npy",
    ]
    data = {}
    for file in preprocessed_files:
        with dvc.api.open(os.path.join("data/preprocessed", file), mode="rb") as f:
            data[file] = np.load(f)
    return data


def calculate_basic_stats(data: NDArray) -> Tuple[float, float, float, float]:
    return np.mean(data), np.std(data), np.min(data), np.max(data)


def plot_random_frames(data: Dict[str, NDArray], num_frames: int = 2) -> None:
    fig, axes = plt.subplots(len(data), num_frames, figsize=(12, 6 * len(data)))

    for i, (name, video) in enumerate(data.items()):
        random_indices = np.random.choice(video.shape[0], num_frames, replace=False)
        for j, idx in enumerate(random_indices):
            ax = axes[i, j] if len(data) > 1 else axes[j]
            ax.imshow(video[idx])
            ax.set_title(f"{name}\nFrame {idx}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("analysis_results/random_frames.png")
    plt.close()


def print_and_save_data_info(data: Dict[str, NDArray]) -> Dict[str, Dict]:
    results = {}
    for name, array in data.items():
        print(f"\nData Info for {name}:")
        info = {
            "Shape": list(array.shape),
            "Data Type": str(array.dtype),
            "Memory Usage": f"{array.nbytes / (1024 * 1024):.2f} MB",
        }
        mean, std, min_val, max_val = calculate_basic_stats(array)
        stats = {
            "Mean": f"{mean:.4f}",
            "Standard Deviation": f"{std:.4f}",
            "Min Value": f"{min_val:.4f}",
            "Max Value": f"{max_val:.4f}",
        }
        info.update(stats)
        results[name] = info

        for key, value in info.items():
            print(f"{key}: {value}")

    return results


def plot_intensity_distribution(data: Dict[str, NDArray]) -> None:
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 5 * len(data)))

    for i, (name, video) in enumerate(data.items()):
        ax = axes[i] if len(data) > 1 else axes
        sns.histplot(video.ravel(), kde=True, ax=ax)
        ax.set_title(f"Pixel Intensity Distribution - {name}")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("analysis_results/intensity_distribution.png")
    plt.close()


def main() -> None:
    try:
        os.makedirs("analysis_results", exist_ok=True)

        data = load_preprocessed_data()
        results = print_and_save_data_info(data)

        with open("analysis_results/data_info.json", "w") as f:
            json.dump(results, f, indent=4)

        plot_random_frames(data)
        plot_intensity_distribution(data)

        print("Analysis completed. Results saved in 'analysis_results' directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
