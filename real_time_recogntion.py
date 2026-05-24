import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import time

MLP_MODEL_PATH  = 'C:/Projects/Sign_language/must_more_blood_be_shed/hand_model_combined.h5'
MLP_LABELS_PATH = 'C:/Projects/Sign_language/must_more_blood_be_shed/label_map_combined.pkl'
CNN_MODEL_PATH  = 'C:/Projects/Sign_language/must_more_blood_be_shed/sign_language_model.h5'

mlp_model = load_model(MLP_MODEL_PATH)
with open(MLP_LABELS_PATH, 'rb') as f:
    le = pickle.load(f)

cnn_model = load_model(CNN_MODEL_PATH)
CNN_IMG_SIZE = (64, 64)

ukrainian_letters = [
    'А','Б','В','Г','Д','Е','Є','Ж','З','I','И',
    'К','Л','М','Н','П','О','Р','С','Т','У','Ф',
    'Х','Ц','Ч','Ш','Ь','Ю','Я'
]

font       = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
font_tiny  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)

USE_CNN = False   # False = MLP (landmarks), True = CNN (image crop)

def draw_text(frame, text, position=(10, 40), f=None, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=f or font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_mode_badge(frame, use_cnn):
    """Top-right badge showing active model."""
    h, w = frame.shape[:2]
    label     = "CNN  (M)" if use_cnn  else "MLP  (M)"
    bg_color  = (180, 60, 0) if use_cnn else (0, 120, 40)   # orange vs green
    badge_w, badge_h = 160, 44
    x0, y0 = w - badge_w - 10, 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), bg_color, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h), (255,255,255), 2)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((x0 + 10, y0 + 6), label, font=font_small, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_word_bar(frame, sentence, current_letter, use_cnn):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h - 110), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if current_letter:
        draw.text((10, h - 100), f"[ {current_letter} ]", font=font, fill=(255, 255, 0))

    sentence_display = "→ " + sentence if sentence else ""
    draw.text((130, h - 100), sentence_display, font=font, fill=(0, 255, 100))

    model_hint = "Model: CNN (press M to switch)" if use_cnn else "Model: MLP (press M to switch)"
    draw.text((10, h - 52), model_hint, font=font_tiny, fill=(160, 200, 255))
    draw.text((10, h - 28), "Hold gesture to confirm  •  No hand = space  •  Bksp = delete  •  C = clear",
              font=font_tiny, fill=(180, 180, 180))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
def predict_cnn(frame, hand_landmarks, frame_w, frame_h):
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

    pred = cnn_model.predict(inp, verbose=0)
    class_id   = int(np.argmax(pred))
    confidence = float(np.max(pred))
    letter     = ukrainian_letters[class_id]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 140, 0), 2)

    return letter, confidence

def predict_mlp(hand_landmarks):
    x_ = [lm.x for lm in hand_landmarks.landmark]
    y_ = [lm.y for lm in hand_landmarks.landmark]
    data_aux = []
    for lm in hand_landmarks.landmark:
        data_aux.append(lm.x - min(x_))
        data_aux.append(lm.y - min(y_))
    X_input = np.array(data_aux).reshape(1, -1)
    prediction = mlp_model.predict(X_input, verbose=0)
    class_id   = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    predicted_label  = le.inverse_transform([class_id])[0]
    predicted_letter = ukrainian_letters[int(predicted_label)]
    return predicted_label, predicted_letter, confidence

mp_hands        = mp.solutions.hands
mp_drawing      = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pred_buffer    = deque(maxlen=10)
confirm_buffer = deque(maxlen=20)
sentence              = ""
last_confirmed_letter = None
no_hand_since         = None
SPACE_DELAY    = 1.2
CONFIRM_RATIO  = 0.75

cap = cv2.VideoCapture(0)

print("Controls:  M = toggle model  |  Bksp = delete  |  C = clear  |  ESC = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    current_letter = None

    if results.multi_hand_landmarks:
        no_hand_since = None

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            if USE_CNN:
                predicted_letter, confidence = predict_cnn(frame, hand_landmarks, w, h)
                if predicted_letter is None:
                    continue
                display_label = predicted_letter
            else:
                predicted_label, predicted_letter, confidence = predict_mlp(hand_landmarks)
                display_label = f"{predicted_label}: {predicted_letter}"

            pred_buffer.append(predicted_letter)
            smoothed_letter = max(set(pred_buffer), key=pred_buffer.count)
            current_letter  = smoothed_letter

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

    frame = draw_mode_badge(frame, USE_CNN)
    frame = draw_word_bar(frame, sentence.strip(), current_letter, USE_CNN)

    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:        
        break
    elif key == ord('m') or key == ord('M'):
        USE_CNN = not USE_CNN
        pred_buffer.clear()
        confirm_buffer.clear()
        last_confirmed_letter = None
        print(f"[Model] Switched to {'CNN' if USE_CNN else 'MLP'}")
    elif key == 8:      
        sentence = sentence[:-1]
    elif key == ord('c') or key == ord('C'):  
        sentence = ""

cap.release()
cv2.destroyAllWindows()