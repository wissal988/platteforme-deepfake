from pathlib import Path
import cv2
import numpy as np
import keras

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "deepfake_video_model.h5"

IMG_SIZE = 224
MAX_SEQ_LENGTH = 15
NUM_FEATURES = 2048
THRESHOLD = 0.5

_video_model = None
_feature_extractor = None

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class CompatDense(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


class CompatGRU(keras.layers.GRU):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        if isinstance(config.get("recurrent_initializer"), dict):
            config["recurrent_initializer"] = "orthogonal"
        if isinstance(config.get("kernel_initializer"), dict):
            config["kernel_initializer"] = "glorot_uniform"
        if isinstance(config.get("bias_initializer"), dict):
            config["bias_initializer"] = "zeros"
        return super().from_config(config)


def load_video_model():
    global _video_model
    if _video_model is None:
        print("LOADING VIDEO MODEL FROM:", MODEL_PATH)
        import tensorflow as tf
        import h5py

        # Reconstruire le modèle manuellement
        input_seq = keras.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), name="input_layer")
        input_mask = keras.Input(shape=(MAX_SEQ_LENGTH,), dtype="bool", name="input_layer_1")

        x = keras.layers.GRU(32, dropout=0.3)(input_seq, mask=input_mask)
        x = keras.layers.Dense(16, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)
        output = keras.layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=[input_seq, input_mask], outputs=output)

        # Charger uniquement les poids
        with h5py.File(MODEL_PATH, "r") as f:
            model.load_weights(MODEL_PATH)

        _video_model = model
        print("Modèle vidéo chargé avec succès")
    return _video_model


def load_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        base_model = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        base_model.trainable = False
        _feature_extractor = base_model
    return _feature_extractor


def crop_center_square(frame):
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def detect_and_crop_face(frame, target_size=(224, 224)):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face = frame[y:y+h, x:x+w]
        if face.size > 0:
            return cv2.resize(face, target_size)

    crop = crop_center_square(frame)
    return cv2.resize(crop, target_size)


def extract_frame_features(frames):
    base_model = load_feature_extractor()
    frames = frames.astype("float32")
    frames = keras.applications.inception_v3.preprocess_input(frames)
    return base_model.predict(frames, verbose=0)


def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = MAX_SEQ_LENGTH

    indices = np.linspace(0, max(total_frames - 1, 0), MAX_SEQ_LENGTH, dtype=int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = detect_and_crop_face(frame)
        frames.append(frame)

    cap.release()
    return np.array(frames)


def predict_video(file_path: str):
    model = load_video_model()
    frames = load_video(file_path)

    if len(frames) == 0:
        raise ValueError("Aucune frame valide")

    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")

    features = extract_frame_features(frames)
    length = min(len(features), MAX_SEQ_LENGTH)

    frame_features[0, :length] = features[:length]
    frame_mask[0, :length] = True

    prob = float(model.predict([frame_features, frame_mask], verbose=0)[0][0])

    label = "FAKE" if prob >= THRESHOLD else "REAL"

    return {
        "label": label,
        "prob_fake": prob,
        "threshold": THRESHOLD,
        "frames_used": int(length),
        "model_name": MODEL_PATH.name
    }