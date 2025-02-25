from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Enable CORS for all origins (Adjust for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URLs for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine',
    'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus',
    'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)',
    'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale',
    'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge',
    'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana',
    'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint',
    'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya',
    'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige',
    'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma',
    'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala'
]  

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))

        # Convert grayscale images to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

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
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Find the predicted class and confidence score
        predicted_index = np.argmax(output_data)
        confidence = float(output_data[predicted_index])

        # Get class label (fallback if index is out of bounds)
        predicted_label = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown"

        return {"predicted_class": predicted_label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")