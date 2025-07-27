import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model
model_path = 'model/plant_disease_model.h5'
model = load_model(model_path)

# Path to test image
img_path = 'test/test_leaf.jpg' #paste your test image path here

# Load and preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # normalize
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])

# Reconstruct class labels from training directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use a dummy ImageDataGenerator to get class indices from dataset folder
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
dummy_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=1,
    class_mode='categorical',
    subset='training'
)

# Get mapping from class index to class name
class_indices = dummy_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

predicted_class_name = index_to_class[predicted_class_index]

# Output result
print(f" Predicted Class: {predicted_class_name}")
print(f" Prediction Probabilities: {predictions[0]}")

# Optional: show the test image
plt.imshow(img)
plt.title(f"Prediction: {predicted_class_name}")
plt.axis('off')
plt.show()
