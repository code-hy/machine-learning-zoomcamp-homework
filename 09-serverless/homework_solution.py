import os
import urllib.request
from io import BytesIO
from PIL import Image  # This library handles image editing
import numpy as np     # This library handles the heavy math
import onnxruntime as ort # This runs the AI model

# --- 1. SETUP: Define where our files and images are ---
MODEL_URL = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx"
MODEL_DATA_URL = "https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle/hair_classifier_v1.onnx.data"
IMAGE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
MODEL_FILENAME = "hair_classifier_v1.onnx"
MODEL_DATA_FILENAME = "hair_classifier_v1.onnx.data"

# Helper function to download a file from the web
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully!")
    else:
        print(f"Found {filename} locally.")

# Helper function to download an image into memory
def download_image(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Helper function to resize the image (Question 2)
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize to the target size (e.g., 200x200)
    img = img.resize(target_size, Image.NEAREST)
    return img

# Helper function to do the Math (Question 3)
def preprocess_imagenet(x):
    # These are magic numbers used by almost all Google/Facebook models
    # They represent the average color and spread of color in the world
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    
    # 1. Squish 0-255 down to 0-1
    x = x / 255.0
    
    # 2. Normalize (Subtract mean, divide by std)
    x = (x - mean) / std
    return x

def main():
    # --- Step 1: Get the Model (both files!) ---
    download_file(MODEL_URL, MODEL_FILENAME)
    download_file(MODEL_DATA_URL, MODEL_DATA_FILENAME)
    
    # Start the "Factory" (Load the model)
    session = ort.InferenceSession(MODEL_FILENAME)
    
    # --- Answer Question 1 ---
    # We ask the model for the name of its output layer
    output_name = session.get_outputs()[0].name
    print(f"\n--- Question 1 ---")
    print(f"The Output Node name is: '{output_name}'")
    
    # --- Answer Question 2 ---
    # We ask the model for the shape of its input layer
    input_node = session.get_inputs()[0]
    input_name = input_node.name
    # shape is usually (Batch_Size, Channels, Height, Width)
    input_shape = input_node.shape 
    print(f"\n--- Question 2 ---")
    print(f"The Model expects input shape: {input_shape}")
    
    # We extract the Height and Width (200, 200)
    target_size = (input_shape[2], input_shape[3])
    print(f"So the target image size is: {target_size}")

    # --- Answer Question 3 ---
    # Download the hair image
    img = download_image(IMAGE_URL)
    # Resize it to 200x200
    img_resized = prepare_image(img, target_size)
    # Convert image to a grid of numbers
    x_raw = np.array(img_resized, dtype='float32')
    
    # Apply the "Math-ifying" (Preprocessing)
    x_preprocessed = preprocess_imagenet(x_raw)
    
    # Look at the very first pixel (Red channel)
    r_value = x_preprocessed[0, 0, 0]
    
    print(f"\n--- Question 3 ---")
    print(f"After math, the first pixel value is: {r_value:.3f}")
    
    # --- Answer Question 4 ---
    # The model expects the data in a specific order: (Channels, Height, Width)
    # But currently it is (Height, Width, Channels). We must swap them.
    x_transposed = x_preprocessed.transpose(2, 0, 1)
    
    # Add a "Batch" dimension. (The model expects a list of photos, even if it's a list of 1)
    input_tensor = x_transposed[np.newaxis, ...]
    
    # RUN THE MODEL!
    outputs = session.run([output_name], {input_name: input_tensor})
    
    # Get the number inside the result
    final_score = outputs[0][0][0]
    
    print(f"\n--- Question 4 ---")
    print(f"The Model predicts: {final_score:.2f}")

if __name__ == "__main__":
    main()
