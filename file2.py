import numpy as np
from tensorflow.keras.preprocessing import image
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models


# Load the trained model
model = models.load_model('maggi_defect_detection_model.h5')

# Path to the image you want to test
image_path = 'path/to/your/image.jpg'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)

# Interpret the prediction
if prediction[0] > 0.5:
    print("Defective Maggi Tablet")
else:
    print("Good Maggi Tablet")
