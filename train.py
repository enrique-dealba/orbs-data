import os

import dvc.api
import numpy as np


def load_preprocessed_data():
    preprocessed_files = [
        "control1_stack_normalized.npy",
        "control2_stack_normalized.npy",
    ]

    data = {}
    for file in preprocessed_files:
        with dvc.api.open(os.path.join("data/preprocessed", file), mode="rb") as f:
            data[file] = np.load(f)
    return data


def main():
    preprocessed_data = load_preprocessed_data()

    for file, data in preprocessed_data.items():
        print(f"{file} shape: {data.shape}")

    # Model goes here


if __name__ == "__main__":
    main()
