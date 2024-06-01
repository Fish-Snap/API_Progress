from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

def tflite_detect_image(model_path, image_path, label_path, min_conf=0.5):
    # Load the label map into memory
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model into memory
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))

    # Normalize pixel values to FLOAT32
    input_data = image_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    best_detection = None
    best_score = 0.0

    # Loop over all detections and find the one with the highest confidence
    for i in range(len(scores)):
        if scores[i] > min_conf and scores[i] > best_score:
            object_name = labels[int(classes[i])]
            best_detection = {'object_name': object_name, 'confidence': float(scores[i]), 'bbox': boxes[i].tolist()}
            best_score = scores[i]

    if best_detection is not None:
        return {"confidence": best_detection['confidence'], "object_name": best_detection['object_name']}
    else:
        return {}

@app.route('/prediction', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded image
        image_path = 'uploaded_image.jpg'
        file.save(image_path)

        # Perform detection on the uploaded image
        detections = tflite_detect_image(model_path='detect_ds14.tflite',
                                         image_path=image_path,
                                         label_path='labels.txt',
                                         min_conf=0.5)

        # Delete the uploaded image
        os.remove(image_path)

        return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
