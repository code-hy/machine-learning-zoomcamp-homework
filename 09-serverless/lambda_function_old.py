
import json
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import onnxruntime as ort

# Global variables - model loaded once per Lambda container
MODEL_PATH = "hair_classifier_empty.onnx"
session = None

def get_session():
    """Lazy load model"""
    global session
    if session is None:
        session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess_image(img_data):
    """Preprocess image for model"""
    # Decode base64 image
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))

    # Prepare image (128x128)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((128, 128), Image.NEAREST)

    # Normalize and standardize
    x = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std

    # Transpose and add batch dimension
    x = x.transpose(2, 0, 1)
    x = x.astype('float32')
    x = np.expand_dims(x, axis=0)

    return x

def lambda_handler(event, context):
    """Lambda function handler"""
    try:
        # Parse request body
        body = json.loads(event['body'])
        image_data = body.get('image')

        if not image_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }

        # Preprocess image
        input_tensor = preprocess_image(image_data)

        # Run inference
        session = get_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        outputs = session.run([output_name], {input_name: input_tensor})
        prediction = float(outputs[0][0][0])

        # Return result
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction,
                'class': 'straight' if prediction < 0.5 else 'curly'
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
