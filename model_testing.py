import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\smvc_project\plant_disease_model_transfer.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ==========================
# LOAD MODEL
# ==========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, len(class_names))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==========================
# IMAGE TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_image(image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class_idx = torch.max(probs, 1)
        return class_names[pred_class_idx.item()], confidence.item()

# ==========================
# MAIN SCRIPT
# ==========================
if __name__ == "__main__":
    # Hardcoded image path
    image_path = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\smvc_project\2a800c84-902e-4aa8-af13-4e020d6f17dd___FREC_Scab 3124.JPG"
    predicted_class, confidence = predict_image(image_path)
    print(f"Predicted Class: {predicted_class} | Confidence: {confidence:.2f}")

    if predicted_class.lower() == "healthy":
        print(" This plant is healthy!")
    else:
        print(f" This plant may have {predicted_class}")

