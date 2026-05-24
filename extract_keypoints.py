import argparse
import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "Data", "default_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand keypoints from a dataset.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        help="Dataset folder containing class subfolders with images.",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("KEYPOINTS_OUTPUT"),
        help="Output pickle path. Defaults to <dataset folder>_keypoints.pickle.",
    )
    return parser.parse_args()


args = parse_args()
DATA_DIR = args.data_dir
OUTPUT_PATH = args.output or f"{DATA_DIR}_keypoints.pickle"

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Dataset folder does not exist: {DATA_DIR}")

import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue 
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print(f"Saved {len(data)} samples to {OUTPUT_PATH}")
