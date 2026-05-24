import argparse
import json
import shutil
import pickle
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
LEGACY_MODELS_DIR = Path("C:/Projects/Sign_language/must_more_blood_be_shed")


def first_existing_path(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def make_legacy_keras_config(value):
    if isinstance(value, dict):
        if (
            value.get("class_name") == "DTypePolicy"
            and isinstance(value.get("config"), dict)
            and "name" in value["config"]
        ):
            return value["config"]["name"]

        if (
            value.get("class_name") == "InputLayer"
            and isinstance(value.get("config"), dict)
            and "batch_shape" in value["config"]
        ):
            value["config"]["batch_input_shape"] = value["config"].pop("batch_shape")

        for key, item in list(value.items()):
            value[key] = make_legacy_keras_config(item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = make_legacy_keras_config(item)

    return value


def load_model_compatible(model_path):
    try:
        return load_model(model_path, compile=False)
    except TypeError as error:
        message = str(error)
        if "batch_shape" not in message and "DTypePolicy" not in message:
            raise

    import h5py

    temp_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        shutil.copy2(model_path, temp_path)
        with h5py.File(temp_path, "r+") as h5_file:
            model_config = h5_file.attrs.get("model_config")
            if model_config is None:
                raise ValueError(f"No model_config found in {model_path}")
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")
            legacy_config = make_legacy_keras_config(json.loads(model_config))
            h5_file.attrs.modify("model_config", json.dumps(legacy_config))

        return load_model(temp_path, compile=False)
    finally:
        temp_path.unlink(missing_ok=True)


MLP_OPTIONS = {
    "K": {
        "name": "K",
        "model_path": MODELS_DIR / "mlp_data.h5",
        "labels_path": MODELS_DIR / "mlp_data_label_map.pkl",
    },
    "O": {
        "name": "O",
        "model_path": MODELS_DIR / "mlp_data_Oles.h5",
        "labels_path": MODELS_DIR / "mlp_data_Oles_label_map.pkl",
    },
    "R": {
        "name": "R",
        "model_path": MODELS_DIR / "mlp_data_roma.h5",
        "labels_path": MODELS_DIR / "mlp_data_roma_label_map.pkl",
    },
    "Combined": {
        "name": "Combined",
        "model_path": first_existing_path(
            MODELS_DIR / "hand_model_combined.h5",
            LEGACY_MODELS_DIR / "hand_model_combined.h5",
        ),
        "labels_path": first_existing_path(
            MODELS_DIR / "label_map_combined.pkl",
            LEGACY_MODELS_DIR / "label_map_combined.pkl",
        ),
    },
}

CNN_OPTION = {
    "name": "General",
    "model_path": first_existing_path(
        MODELS_DIR / "sign_language_model.h5",
        LEGACY_MODELS_DIR / "sign_language_model.h5",
        BASE_DIR / "sign_language_model.h5",
    ),
    "normalize": False,
}

parser = argparse.ArgumentParser(description="Real-time sign language recognition")
parser.add_argument("--model-type", choices=["MLP", "CNN"], default="MLP")
parser.add_argument("--person", choices=[*MLP_OPTIONS, "Custom"], default="Combined")
parser.add_argument("--model-name", default="Custom")
parser.add_argument("--mlp-model-path", type=Path)
parser.add_argument("--mlp-labels-path", type=Path)
parser.add_argument("--cnn-model-path", type=Path)
parser.add_argument("--cnn-labels-path", type=Path)
args = parser.parse_args()

if args.model_type == "MLP" and args.person == "Custom":
    if not args.mlp_model_path or not args.mlp_labels_path:
        parser.error("--person Custom requires --mlp-model-path and --mlp-labels-path")
    MLP_OPTIONS["Custom"] = {
        "name": args.model_name,
        "model_path": args.mlp_model_path,
        "labels_path": args.mlp_labels_path,
    }

if args.model_type == "CNN" and args.cnn_model_path:
    CNN_OPTION["name"] = args.model_name
    CNN_OPTION["model_path"] = args.cnn_model_path
    CNN_OPTION["labels_path"] = args.cnn_labels_path
    CNN_OPTION["normalize"] = True

MODEL_TYPE = args.model_type
ACTIVE_MLP_KEY = args.person

model_cache = {}
label_cache = {}
CNN_IMG_SIZE = (64, 64)

ukrainian_letters = [
    "\u0410", "\u0411", "\u0412", "\u0413", "\u0414", "\u0415", "\u0404", "\u0416", "\u0417", "I", "\u0418",
    "\u041a", "\u041b", "\u041c", "\u041d", "\u041f", "\u041e", "\u0420", "\u0421", "\u0422", "\u0423", "\u0424",
    "\u0425", "\u0426", "\u0427", "\u0428", "\u042c", "\u042e", "\u042f",
]

font       = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
font_tiny  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)


def get_mlp_model(model_key):
    if model_key not in model_cache:
        option = MLP_OPTIONS[model_key]
        model_cache[model_key] = load_model_compatible(option["model_path"])
        with open(option["labels_path"], "rb") as f:
            label_cache[model_key] = pickle.load(f)
    return model_cache[model_key], label_cache[model_key]


def get_cnn_model():
    cache_key = "CNN-General"
    if cache_key not in model_cache:
        model_cache[cache_key] = load_model_compatible(CNN_OPTION["model_path"])
        labels_path = CNN_OPTION.get("labels_path")
        if labels_path:
            with open(labels_path, "rb") as f:
                label_cache[cache_key] = pickle.load(f)
    return model_cache[cache_key]


def active_model_name():
    if MODEL_TYPE == "CNN":
        return f"CNN / {CNN_OPTION['name']}"
    return f"MLP / {MLP_OPTIONS[ACTIVE_MLP_KEY]['name']}"


def label_to_letter(label):
    try:
        index = int(label)
    except (TypeError, ValueError):
        return str(label)
    return ukrainian_letters[index] if 0 <= index < len(ukrainian_letters) else str(label)


def draw_text(frame, text, position=(10, 40), f=None, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=f or font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_mode_badge(frame):
    h, w = frame.shape[:2]
    label = active_model_name()
    bg_color = (180, 60, 0) if MODEL_TYPE == "CNN" else (0, 120, 40)
    badge_w, badge_h = 230, 44
    x0, y0 = w - badge_w - 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), bg_color, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), (255, 255, 255), 2)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((x0 + 10, y0 + 6), label, font=font_small, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_word_bar(frame, sentence, current_letter):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h - 110), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if current_letter:
        draw.text((10, h - 100), f"[ {current_letter} ]", font=font, fill=(255, 255, 0))

    sentence_display = "-> " + sentence if sentence else ""
    draw.text((130, h - 100), sentence_display, font=font, fill=(0, 255, 100))

    model_hint = f"Model: {active_model_name()}"
    draw.text((10, h - 52), model_hint, font=font_tiny, fill=(160, 200, 255))
    draw.text(
        (10, h - 28),
        "Hold gesture to confirm  *  No hand = space  *  Bksp = delete  *  C = clear",
        font=font_tiny,
        fill=(180, 180, 180),
    )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def predict_cnn(frame, hand_landmarks, frame_w, frame_h):
    cnn_model = get_cnn_model()
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    pad = 0.15
    x1 = max(0, int((min(x_coords) - pad) * frame_w))
    y1 = max(0, int((min(y_coords) - pad) * frame_h))
    x2 = min(frame_w, int((max(x_coords) + pad) * frame_w))
    y2 = min(frame_h, int((max(y_coords) + pad) * frame_h))

    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    crop = frame[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, CNN_IMG_SIZE)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(crop_rgb, axis=0).astype(np.float32)
    if CNN_OPTION.get("normalize"):
        inp /= 255.0

    pred = cnn_model.predict(inp, verbose=0)
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))
    label_encoder = label_cache.get("CNN-General")
    if label_encoder is not None:
        label = label_encoder.inverse_transform([class_id])[0]
        letter = label_to_letter(label)
    else:
        letter = label_to_letter(class_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)

    return letter, confidence


def predict_mlp(hand_landmarks):
    mlp_model, le = get_mlp_model(ACTIVE_MLP_KEY)
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    data_aux = []
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))
    X_input = np.array(data_aux).reshape(1, -1)
    prediction = mlp_model.predict(X_input, verbose=0)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label = le.inverse_transform([class_id])[0]
    predicted_letter = label_to_letter(predicted_label)
    return predicted_label, predicted_letter, confidence


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pred_buffer = deque(maxlen=10)
confirm_buffer = deque(maxlen=20)
sentence = ""
last_confirmed_letter = None
no_hand_since = None
SPACE_DELAY = 1.2
CONFIRM_RATIO = 0.75

cap = cv2.VideoCapture(0)

print(f"Model: {active_model_name()}")
print("Controls:  Bksp = delete  |  C = clear  |  ESC = quit")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_letter = None

    if results.multi_hand_landmarks:
        no_hand_since = None

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            if MODEL_TYPE == "CNN":
                predicted_letter, confidence = predict_cnn(frame, hand_landmarks, w, h)
                if predicted_letter is None:
                    continue
                display_label = predicted_letter
            else:
                predicted_label, predicted_letter, confidence = predict_mlp(hand_landmarks)
                display_label = f"{predicted_label}: {predicted_letter}"

            pred_buffer.append(predicted_letter)
            smoothed_letter = max(set(pred_buffer), key=pred_buffer.count)
            current_letter = smoothed_letter

            confirm_buffer.append(smoothed_letter)

            if len(confirm_buffer) == confirm_buffer.maxlen:
                most_common = max(set(confirm_buffer), key=confirm_buffer.count)
                ratio = confirm_buffer.count(most_common) / len(confirm_buffer)
                if ratio >= CONFIRM_RATIO and most_common != last_confirmed_letter:
                    sentence += most_common
                    last_confirmed_letter = most_common
                    confirm_buffer.clear()

            info = f"{display_label}  ({confidence:.2f})"
            frame = draw_text(frame, info, position=(10, 10))

    else:
        pred_buffer.clear()
        confirm_buffer.clear()
        last_confirmed_letter = None

        if no_hand_since is None:
            no_hand_since = time.time()
        elif time.time() - no_hand_since >= SPACE_DELAY:
            if sentence and not sentence.endswith(" "):
                sentence += " "
            no_hand_since = time.time() + 9999

    frame = draw_mode_badge(frame)
    frame = draw_word_bar(frame, sentence.strip(), current_letter)

    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 8:
        sentence = sentence[:-1]
    elif key == ord("c") or key == ord("C"):
        sentence = ""

cap.release()
cv2.destroyAllWindows()
