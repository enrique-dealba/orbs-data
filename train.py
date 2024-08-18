import dvc.api
import numpy as np


def load_split_data():
    with dvc.api.open("data/split/train_data.npy", mode="rb") as f:
        train_data = np.load(f)
    with dvc.api.open("data/split/val_data.npy", mode="rb") as f:
        val_data = np.load(f)
    return train_data, val_data


def main():
    train_data, val_data = load_split_data()

    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # TODO: model training
    # model = train_model(train_data, val_data)
    # model.save(...)


if __name__ == "__main__":
    main()
