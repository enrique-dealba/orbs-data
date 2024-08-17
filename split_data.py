import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import dvc.api

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

def split_and_save_data(data, test_size=0.2, random_state=42):
    split_data = {}
    for name, array in data.items():
        # Assuming the first dimension is the number of frames
        n_frames = array.shape[0]
        indices = np.arange(n_frames)
        
        train_indices, val_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        split_data[name] = {
            'train': array[train_indices],
            'val': array[val_indices]
        }
    
    # Save the split data
    os.makedirs('data/split', exist_ok=True)
    for name, splits in split_data.items():
        np.save(f'data/split/{name[:-4]}_train.npy', splits['train'])
        np.save(f'data/split/{name[:-4]}_val.npy', splits['val'])
    
    # Save the split indices for reproducibility
    split_indices = {
        name: {
            'train': train_indices.tolist(),
            'val': val_indices.tolist()
        } for name, (train_indices, val_indices) in zip(
            data.keys(),
            [(train_test_split(np.arange(arr.shape[0]), test_size=test_size, random_state=random_state)) for arr in data.values()]
        )
    }
    
    with open('data/split/split_indices.json', 'w') as f:
        json.dump(split_indices, f)

def load_split_data():
    split_data = {}
    for file in os.listdir('data/split'):
        if file.endswith('.npy'):
            name, split_type = os.path.splitext(file)[0].rsplit('_', 1)
            if name not in split_data:
                split_data[name] = {}
            split_data[name][split_type] = np.load(os.path.join('data/split', file))
    return split_data

def main():
    data = load_preprocessed_data()
    split_and_save_data(data)
    print("Data has been split and saved in 'data/split' directory.")

if __name__ == "__main__":
    main()
