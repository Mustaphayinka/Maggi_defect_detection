import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('maggi_defect_detection_model.h5')

# Define image dimensions
img_width, img_height = 150, 150

# Define a function to process the uploaded image
def process_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        # Check if the file is allowed
        if file:
            # Save the uploaded file
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            # Process the image and make prediction
            img_array = process_image(file_path)
            prediction = model.predict(img_array)
            result = "Defective Maggi Tablet" if prediction[0] > 0.5 else "Good Maggi Tablet"
            return render_template('index.html', message=result)
    return render_template('index.html', message='')

if __name__ == '__main__':
    app.run(debug=True)
