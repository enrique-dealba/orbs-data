import os

import cv2
import dvc.api
import numpy as np


def load_avi_to_numpy(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def normalize_video_data(video_data):
    return video_data.astype(np.float32) / 255.0


def main():
    avi_files = [
        "control1_stack.avi",
        "control2_stack.avi",
    ]

    input_dir = "data/raw_avis"
    output_dir = "data/preprocessed"
    os.makedirs(output_dir, exist_ok=True)

    for avi_file in avi_files:
        # Use DVC to get the path of the input file
        with dvc.api.open(os.path.join(input_dir, avi_file)) as f:
            input_path = f.name

        # Load and preprocess the data
        print(f"Processing {avi_file}...")
        video_data = load_avi_to_numpy(input_path)
        normalized_data = normalize_video_data(video_data)

        # Save the preprocessed data
        output_filename = os.path.splitext(avi_file)[0] + "_normalized.npy"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, normalized_data)
        print(f"Saved preprocessed data to {output_path}")


if __name__ == "__main__":
    main()
