import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = os.path.join(os.getcwd(), 'model', 'plant_disease_model.h5')
model = load_model(MODEL_PATH)

class_names = [
    'Apple___Black_rot',
    'Apple___Scab',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust',
    'Corn_(maize)___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___healthy'
]


def predict_disease(img_path):
    # Resize image to match model's expected input size
    img = image.load_img(img_path, target_size=(128, 128))  # ← FIXED HERE
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]

    if len(prediction) != len(class_names):
        raise ValueError(f"Model returned {len(prediction)} outputs, but {len(class_names)} class names are defined.")

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {'class': predicted_class, 'confidence': confidence}
