from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.app import App
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.filemanager import MDFileManager
from kivymd.app import MDApp
from PIL import Image as PILImage
import torch
import torchvision.transforms as transforms
from timm import create_model
import numpy as np
import glob
import className  # Import scientific names and descriptions

# List of class names
class_names = [
    "Aloevera", "Amla", "Amruthaballi", "Arali", "Arive-Dantu", "Astma_weed", "Badipala", "Balloon_Vine", "Bamboo",
    "Basale", "Beans", "Betel", "Bhrami", "Bringaraja", "Caricature", "Castor", "Catharanthus", "Chakte", "Chilly",
    "Citron lime (herelikai)", "Coffee", "Common rue(naagdalli)", "Coriender", "Crape_Jasmine", "Curry", "Doddpathre",
    "Drumstick", "Ekka", "Eucalyptus", "Fenugreek", "Ganigale", "Ganike", "Gasagase", "Ginger", "Globe Amarnath",
    "Guava", "Henna", "Hibiscus", "Honge", "Indian_Beech", "Indian_Mustard", "Insulin", "Jackfruit",
    "Jamaica_Cherry-Gasagase", "Jamun", "Jasmine", "Kambajala", "Karanda", "Kasambruga", "Kohlrabi", "Lantana",
    "Lemon", "Lemongrass", "Malabar_Nut", "Malabar_Spinach", "Mango", "Marigold", "Mexican_Mint", "Mint", "Neem",
    "Nelavembu", "Nerale", "Nooni", "Oleander", "Onion", "Padri", "Palak(Spinach)", "Papaya", "Parijata", "Parijatha",
    "Pea", "Peepal", "Pepper", "Pomoegranate", "Pumpkin", "Raddish", "Rasna", "Rose", "Rose_apple",
    "Roxburgh_fig", "Sampige", "Sandalwood", "Sapota", "Seethaashoka", "Seethapala", "Spinach1", "Tamarind", "Taro",
    "Tecoma", "Thumbe", "Tomato", "Tulsi", "Turmeric", "ashoka", "camphor", "kamakasturi", "kepala"
]
CLASS_NAMES = np.array(class_names)
device = "cpu"
model_paths = sorted(glob.glob("best_swin_fold*.pth"))
num_classes = 97

# ✅ Load Multiple Swin Transformer Models
models = []
for model_path in model_paths:
    model = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    models.append(model)

# ✅ Preprocess image for model
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = PILImage.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# ✅ Ensemble Prediction
def ensemble_predict(models, image_path):
    image = process_image(image_path)
    outputs = []

    with torch.no_grad():
        for model in models:
            outputs.append(model(image).cpu())  # Get predictions

    avg_output = torch.mean(torch.stack(outputs), dim=0)  # Average predictions
    predicted_index = torch.argmax(avg_output).item()
    predicted_class_name = CLASS_NAMES[predicted_index]
    return predicted_class_name, predicted_index  # Return both class name and index

    
# ✅ Kivy UI with all four information fields
KV = """
BoxLayout:
    orientation: 'vertical'
    spacing: 10
    padding: 20

    Image:
        id: img
        source: ''
        size_hint: (1, 0.4)
        allow_stretch: True
        keep_ratio: True

    MDRaisedButton:
        text: "Upload Image"
        on_release: app.open_file_manager()
        pos_hint: {"center_x": 0.5}
        size_hint: (0.6, None)
        height: 50

    MDRaisedButton:
        text: "Predict"
        on_release: app.predict()
        pos_hint: {"center_x": 0.5}
        size_hint: (0.6, None)
        height: 50

    Label:
        id: result
        text: "Prediction: "
        font_size: 18
        color: 0, 0, 0, 1
        size_hint: (1, 0.15)
        halign: "center"
        text_size: self.size

    Label:
        id: scientific_name
        text: "Scientific Name: "
        font_size: 18
        color: 0, 0, 0, 1
        size_hint: (1, 0.15)
        halign: "center"
        text_size: self.size

    Label:
        id: plant_info
        text: "Information: "
        font_size: 16
        color: 0, 0, 0, 1
        size_hint: (1, 0.2)
        halign: "center"
        text_size: self.size

    Label:
        id: plant_link
        text: "Link: "
        font_size: 16
        color: 0, 0, 1, 1  # Blue color for link
        size_hint: (1, 0.1)
        halign: "center"
        text_size: self.size
"""

class LeafSnapApp(MDApp):
    def build(self):
        self.file_manager = MDFileManager(exit_manager=self.exit_manager, select_path=self.select_file)
        return Builder.load_string(KV)

    def open_file_manager(self):
        self.file_manager.show('/')  # Open file manager

    def select_file(self, file_path):
        self.root.ids.img.source = file_path
        self.image_path = file_path
        self.exit_manager()

    def exit_manager(self, *args):
        self.file_manager.close()

    def predict(self):
        if not hasattr(self, 'image_path'):
            self.root.ids.result.text = "Please upload an image first."
            return

        predicted_class_name, predicted_index = ensemble_predict(models, self.image_path)

        # ✅ Display all information
        self.root.ids.result.text = f"Prediction: {predicted_class_name}"
        self.root.ids.scientific_name.text = f"Scientific Name: {className.scientific_names[predicted_index]}"
        self.root.ids.plant_info.text = f"Information: {className.plants_info[predicted_index]}"
        self.root.ids.plant_link.text = f"Link: {className.link[predicted_index]}"

if __name__ == "__main__":
    LeafSnapApp().run()
