import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings("ignore")

# # Define paths to your dataset
# dataset_dir = 'Sodiq'
# train_dir = os.path.join(dataset_dir, 'Well')
# val_dir = os.path.join(dataset_dir, 'Well')

# # Create train and validation directories if not already present
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)

# # Split dataset into train and validation sets (80-20 split)
# good_images = os.listdir(os.path.join(dataset_dir, 'Well'))
# bad_images = os.listdir(os.path.join(dataset_dir, 'Bad'))

# # Shuffle images to ensure randomness
# random.shuffle(good_images)
# random.shuffle(bad_images)

# # Define split ratio
# split_ratio = 0.8

# # Split images into train and validation sets
# good_train_size = int(len(good_images) * split_ratio)
# bad_train_size = int(len(bad_images) * split_ratio)

# # Move images to train directory
# for img in good_images[:good_train_size]:
#     src = os.path.join(dataset_dir, 'Well', img)
#     dst = os.path.join(train_dir, 'Well', img)
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     shutil.copy(src, dst)

# for img in bad_images[:bad_train_size]:
#     src = os.path.join(dataset_dir, 'Bad', img)
#     dst = os.path.join(train_dir, 'Bad', img)
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     shutil.copy(src, dst)

# # Move remaining images to validation directory
# for img in good_images[good_train_size:]:
#     src = os.path.join(dataset_dir, 'Well', img)
#     dst = os.path.join(val_dir, 'Well', img)
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     shutil.copy(src, dst)

# for img in bad_images[bad_train_size:]:
#     src = os.path.join(dataset_dir, 'Bad', img)
#     dst = os.path.join(val_dir, 'Bad', img)
#     os.makedirs(os.path.dirname(dst), exist_ok=True)
#     shutil.copy(src, dst)

# # Define image dimensions and batch size
# img_width, img_height = 150, 150
# batch_size = 32

# # Define CNN architecture
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Create data generators
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')
# val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

# # Train the model
# history = model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=10, validation_data=val_generator, validation_steps=val_generator.samples // batch_size)

# # Evaluate the model
# val_loss, val_acc = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
# print('Validation accuracy:', val_acc)

# # Save the model
# model.save('maggi_defect_detection_model.h5')

import numpy as np
from tensorflow.keras.preprocessing import image


# Load the trained model
model = models.load_model('maggi_defect_detection_model.h5')

# Path to the image you want to test
image_path = r"C:\Users\user\OneDrive\Desktop\New folder\Sodiq\Sodiq\Well\Well\IMG_20240323_152637_879.jpg"

# Load and preprocess the image
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0


# Make prediction
prediction = model.predict(img_array)

# Interpret the prediction
if prediction[0] > 0.5:
    print("Defective Maggi Tablet")
else:
    print("Good Maggi Tablet")
