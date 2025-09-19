import streamlit as st
import tempfile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import json


import difflib  # for closest matching

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\smvc_project\plant_disease_model_transfer.pth"
JSON_PATH = r"C:\Users\thenn\OneDrive\thenn\OneDrive\Desktop\smvc_project\expanded_plant_disease_recommendations.json"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# CLASS NAMES
# ==============================
class_names = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___Late_blight",
    "Potato___healthy","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot","Tomato___Early_blight",
    "Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

# ==============================
# LOAD DISEASE JSON
# ==============================
with open(JSON_PATH, "r") as f:
    disease_info = json.load(f)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==============================
# IMAGE TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================
# PREDICTION FUNCTION
# ==============================


def predict_image(image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        if pred_class_idx >= len(class_names):
            return "Unknown", 0.0
        confidence = probs[0, pred_class_idx].item()
        return class_names[pred_class_idx], confidence

# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title("🌱 Plant Disease Detection System")

if "image_file_path" not in st.session_state:
    st.session_state.image_file_path = None

# --- Upload Image ---
st.subheader("Upload Plant Leaf Image")
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(uploaded_file.getbuffer())
    st.session_state.image_file_path = temp_file.name
    st.image(temp_file.name, caption="Selected Image", use_container_width=True)

# --- Prediction ---
if st.session_state.image_file_path:
    st.write("---")
    st.subheader("Prediction Result")
    predicted_class, confidence = predict_image(st.session_state.image_file_path)

    st.write(f"**Detected Disease:** {predicted_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    if "healthy" in predicted_class.lower():
        st.success("This plant is healthy! ✅")
    else:
        st.warning(f"This plant may have {predicted_class} ⚠️")

    # --- Recommendations ---
    predicted_class_clean = predicted_class.strip()

    # Try exact match, then case-insensitive match, then closest match
    info = disease_info.get(predicted_class_clean)
    if info is None:
        for key in disease_info.keys():
            if key.lower().strip() == predicted_class_clean.lower():
                info = disease_info[key]
                predicted_class_clean = key
                break

    if info is None:
        closest = difflib.get_close_matches(predicted_class_clean, disease_info.keys(), n=1, cutoff=0.6)
        if closest:
            info = disease_info[closest[0]]
            predicted_class_clean = closest[0]

    if info:
        st.subheader("🌿 Disease Management Tips")
        for key, title in [
            ("preventive", "Preventive Measures"),
            ("organic", "Organic Treatments"),
            ("chemical", "Chemical Treatments"),
            ("notes", "Notes")
        ]:
            if key in info and info[key]:
                st.markdown(f"**{title}:**")
                if isinstance(info[key], list):
                    for item in info[key]:
                        st.write(f"- {item}")
                else:
                    st.write(info[key])
    else:
        st.error(f"No recommendations found for: **{predicted_class_clean}**")
        with st.expander(" Debug: Available JSON Keys"):
            st.write(list(disease_info.keys()))
else:
    st.info("Upload or capture an image to see predictions.")
