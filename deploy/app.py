from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

model = tf.keras.models.load_model('D:\Graduation project\deploy\imageclassifier.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Read the image file and convert it to a NumPy array
    image = Image.open(file)
    image = image.resize((256, 256))  # Resize the image if required
    image = np.array(image) / 255.0  # Normalize the image

    # Convert the image to grayscale if necessary
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = np.mean(image, axis=2)

    # Convert the grayscale image to RGB
    image_rgb = np.stack((image,) * 3, axis=-1)

    # Convert the image data type to float32
    image_rgb = image_rgb.astype(np.float32)

    # Add an extra dimension to the image array
    image_rgb = np.expand_dims(image_rgb, axis=0)

    # Perform inference using the loaded model
    predictions = model.predict(image_rgb)

    if predictions > 0.5:
        return jsonify({'class_label': 'not fractured'})
    else:
        return jsonify({'class_label': 'fractured'})

@app.route('/', methods=['POST'])
def index():
    return 'Hello, world!'

if __name__ == '__main__':
    app.run(debug=True)
