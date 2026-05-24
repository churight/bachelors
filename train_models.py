import argparse
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "Data"
PHOTOS_ROOT = DATA_ROOT / "Photos"
MODELS_DIR = PROJECT_DIR / "models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP and CNN sign-language models.")
    parser.add_argument("--data-dir", default=PHOTOS_ROOT / "default_dataset", type=Path)
    parser.add_argument("--keypoints", type=Path, help="Keypoints pickle. Defaults to Data/<dataset>_keypoints.pickle.")
    parser.add_argument("--model-type", choices=["MLP", "CNN", "Both"], default="MLP")
    parser.add_argument("--name", help="Model output name. Defaults to dataset folder name.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=64)
    return parser.parse_args()


def safe_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "custom"


def label_to_display(label):
    ukrainian_letters = [
        "\u0410", "\u0411", "\u0412", "\u0413", "\u0414", "\u0415", "\u0404", "\u0416", "\u0417", "I", "\u0418",
        "\u041a", "\u041b", "\u041c", "\u041d", "\u041f", "\u041e", "\u0420", "\u0421", "\u0422", "\u0423", "\u0424",
        "\u0425", "\u0426", "\u0427", "\u0428", "\u042c", "\u042e", "\u042f",
    ]
    try:
        index = int(label)
    except (TypeError, ValueError):
        return str(label)
    return ukrainian_letters[index] if 0 <= index < len(ukrainian_letters) else str(label)


def save_label_encoder(label_encoder, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(label_encoder, f)


def split_stratify_labels(encoded_labels, test_size=0.2):
    class_count = len(np.unique(encoded_labels))
    test_count = math.ceil(len(encoded_labels) * test_size)
    train_count = len(encoded_labels) - test_count
    counts = np.bincount(encoded_labels)
    if min(counts) < 2 or test_count < class_count or train_count < class_count:
        return None
    return encoded_labels


def normalize_keypoint_samples(samples, labels):
    lengths = Counter(len(sample) for sample in samples)
    if not lengths:
        raise ValueError("No keypoint samples found.")

    expected_length, keep_count = lengths.most_common(1)[0]
    filtered_samples = []
    filtered_labels = []
    skipped = 0

    for sample, label in zip(samples, labels):
        if len(sample) == expected_length:
            filtered_samples.append(sample)
            filtered_labels.append(label)
        else:
            skipped += 1

    if skipped:
        print(
            f"Skipped {skipped} keypoint samples with non-{expected_length} feature length "
            f"(length counts: {dict(sorted(lengths.items()))})"
        )

    return np.asarray(filtered_samples, dtype=np.float32), np.asarray(filtered_labels)


def build_mlp(input_shape, class_count):
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(input_shape,)),
            Dropout(0.25),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_mlp(keypoints_path, output_prefix, epochs, batch_size):
    if not keypoints_path.exists():
        raise FileNotFoundError(f"Keypoints file does not exist: {keypoints_path}")

    with open(keypoints_path, "rb") as f:
        dataset = pickle.load(f)

    data, labels = normalize_keypoint_samples(dataset["data"], dataset["labels"])
    if len(data) == 0:
        raise ValueError(f"No keypoint samples found in {keypoints_path}")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    y = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

    stratify = split_stratify_labels(encoded_labels)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = build_mlp(data.shape[1], len(label_encoder.classes_))
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_path = output_prefix.with_name(f"{output_prefix.name}_mlp.h5")
    labels_path = output_prefix.with_name(f"{output_prefix.name}_mlp_label_map.pkl")
    history_path = output_prefix.with_name(f"{output_prefix.name}_mlp.history.json")

    model.save(model_path)
    save_label_encoder(label_encoder, labels_path)
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    print(f"Saved MLP model: {model_path}")
    print(f"Saved MLP labels: {labels_path}")
    print(f"MLP validation accuracy: {accuracy:.4f}")


def iter_image_paths(data_dir):
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path, class_dir.name


def load_cnn_dataset(data_dir, img_size):
    images = []
    labels = []
    for image_path, label in iter_image_paths(data_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.resize(image, (img_size, img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image.astype(np.float32) / 255.0)
        labels.append(label)

    if not images:
        raise ValueError(f"No training images found in {data_dir}")

    return np.asarray(images, dtype=np.float32), np.asarray(labels)


def build_cnn(img_size, class_count):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(data_dir, output_prefix, epochs, batch_size, img_size):
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")

    images, labels = load_cnn_dataset(data_dir, img_size)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    y = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

    stratify = split_stratify_labels(encoded_labels)
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = build_cnn(img_size, len(label_encoder.classes_))
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model_path = output_prefix.with_name(f"{output_prefix.name}_cnn.h5")
    labels_path = output_prefix.with_name(f"{output_prefix.name}_cnn_label_map.pkl")
    history_path = output_prefix.with_name(f"{output_prefix.name}_cnn.history.json")

    model.save(model_path)
    save_label_encoder(label_encoder, labels_path)
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    print(f"Saved CNN model: {model_path}")
    print(f"Saved CNN labels: {labels_path}")
    print(f"CNN validation accuracy: {accuracy:.4f}")


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    keypoints_path = (args.keypoints or DATA_ROOT / f"{data_dir.name}_keypoints.pickle").resolve()
    model_name = safe_name(args.name or data_dir.name)
    MODELS_DIR.mkdir(exist_ok=True)
    output_prefix = MODELS_DIR / f"custom_{model_name}"

    print(f"Dataset: {data_dir}")
    print(f"Output prefix: {output_prefix}")

    if args.model_type in ("MLP", "Both"):
        train_mlp(keypoints_path, output_prefix, args.epochs, args.batch_size)
    if args.model_type in ("CNN", "Both"):
        train_cnn(data_dir, output_prefix, args.epochs, args.batch_size, args.img_size)


if __name__ == "__main__":
    main()
