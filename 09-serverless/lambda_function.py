import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image

# Initialize the model
# NOTE: The homework says the model inside the docker image is named 'hair_classifier_empty.onnx'
MODEL_NAME = "hair_classifier_empty.onnx"
session = ort.InferenceSession(MODEL_NAME)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
target_size = (200, 200)

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(x):
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    x = x / 255.0
    x = (x - mean) / std
    return x

def predict(url):
    img = download_image(url)
    img_prep = prepare_image(img, target_size)
    x = np.array(img_prep, dtype='float32')
    x = preprocess(x)
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis, ...]
    
    result = session.run([output_name], {input_name: x})
    return float(result[0][0])

def lambda_handler(event, context):
    url = event.get('url')
    result = predict(url)
    return {'prediction': result}