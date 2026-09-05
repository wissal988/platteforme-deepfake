from pathlib import Path
import numpy as np
import keras
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "deepfake_image_model.h5"

IMG_HEIGHT = 128
IMG_WIDTH = 128
THRESHOLD = 0.5

_image_model = None


class CompatDense(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


def load_image_model():
    global _image_model
    if _image_model is None:
        _image_model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"Dense": CompatDense},
            safe_mode=False,
        )
    return _image_model


def predict_image(file_path: str):
    model = load_image_model()

    img = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model.predict(img_array, verbose=0)[0][0])

    if prob >= THRESHOLD:
        label = "REAL"
        prob_real = prob
        prob_fake = 1 - prob
    else:
        label = "FAKE"
        prob_real = prob
        prob_fake = 1 - prob

    return {
        "label": label,
        "prob_fake": float(prob_fake),
        "prob_real": float(prob_real),
        "score": float(prob),
        "threshold": THRESHOLD,
        "model_name": MODEL_PATH.name
    }