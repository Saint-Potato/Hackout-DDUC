from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Resize image to match model input shape
    input_shape = input_details[0]['shape']
    image = image.resize((input_shape[1], input_shape[2]))  # Assume (1, height, width, channels)

    # Convert to NumPy array and preprocess
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return {"prediction": output_data.tolist()}
