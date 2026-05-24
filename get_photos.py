import argparse
import os

import cv2


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "Data", "default_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect sign-language photos from webcam.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        help="Folder where class folders with captured photos will be saved.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=int(os.environ.get("NUMBER_OF_CLASSES", 29)),
        help="Number of gesture classes to collect.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=int(os.environ.get("DATASET_SIZE", 100)),
        help="Number of images to collect for each class.",
    )
    return parser.parse_args()


args = parse_args()
DATA_DIR = args.data_dir
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = args.classes
dataset_size = args.dataset_size

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from webcam.")
        cv2.putText(frame, 'Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read frame from webcam.")
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
