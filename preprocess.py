import os
from typing import List, Tuple

import cv2
import numpy as np


def load_avi_to_numpy(file_path: str) -> np.ndarray:
    """Loads AVIs and converts to numpy array."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {file_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.array(frames)


def normalize_video_data(video_data: np.ndarray) -> np.ndarray:
    """Normalizes video data to range [0, 1]."""
    return video_data.astype(np.float32) / 255.0


def preprocess_avi_files(file_paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """Preprocess multiple AVI files."""
    preprocessed_data = []
    for file_path in file_paths:
        try:
            video_data = load_avi_to_numpy(file_path)
            normalized_data = normalize_video_data(video_data)
            file_name = os.path.basename(file_path)
            preprocessed_data.append((file_name, normalized_data))
        except (FileNotFoundError, ValueError) as e:
            print(f"Error processing {file_path}: {str(e)}")
    return preprocessed_data


def main():
    avi_files = [
        "/control1_stack.avi",
        "/control2_stack.avi",
    ]

    preprocessed_videos = preprocess_avi_files(avi_files)

    for file_name, video_data in preprocessed_videos:
        print(
            f"Preprocessed {file_name}: Shape {video_data.shape}, Data range [{video_data.min()}, {video_data.max()}]"
        )


if __name__ == "__main__":
    main()
