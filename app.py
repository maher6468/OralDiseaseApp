# ============================================
# Import Libraries
# ============================================
import pandas as pd
import json

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet50


# ============================================
# Configure Streamlit Page
# ============================================

st.set_page_config(
    page_title="Oral Disease Classification",
    page_icon="🦷",
    layout="centered"
)
# ============================================
# Sidebar
# ============================================

st.sidebar.title("🦷 Oral Disease Classifier")

st.sidebar.markdown("---")

st.sidebar.write("**Model:** ResNet50 (Fine-Tuned)")

st.sidebar.write("**Number of Classes:** 6")

st.sidebar.write("**Framework:** PyTorch")

st.sidebar.write("**Frontend:** Streamlit")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload a clear oral image for better prediction."
)

# ============================================
# Load Class Names
# ============================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)


# ============================================
# Build ResNet50 Model
# ============================================

model = resnet50(weights=None)

num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, NUM_CLASSES)
)


# ============================================
# Load Trained Weights
# ============================================

model.load_state_dict(
    torch.load(
        "resnet50_finetuned.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


# ============================================
# Image Transform
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================
# Application Title
# ============================================
# ============================================
# Disease Information Dictionary
# ============================================

disease_info = {

    "Calculus":
    "Dental calculus is hardened plaque that forms on teeth.",

    "Caries":
    "Dental caries is tooth decay caused by bacteria.",

    "Gingivitis":
    "Inflammation of the gums due to plaque accumulation.",

    "Hypodontia":
    "A congenital condition where one or more teeth are missing.",

    "Mouth_Ulcer":
    "Painful sores that appear inside the mouth.",

    "Tooth_Discoloration":
    "Abnormal tooth color caused by stains or diseases."

}
st.title("🦷 Oral Disease Classification")

st.success("✅ ResNet50 model loaded successfully!")

st.write(
    "Upload an oral image and let the AI model predict the disease."
)


# ============================================
# Upload Image
# ============================================

uploaded_file = st.file_uploader(
    "Choose an oral image",
    type=["jpg", "jpeg", "png"]
)


# ============================================
# Prediction Section
# ============================================

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    # Get predicted class
    predicted_class = class_names[predicted.item()]

    # Display prediction
    st.success(
        f"Prediction: {predicted_class}"
    )

    confidence_score = confidence.item() * 100
    if confidence_score < 50:

        st.warning(
        "Low confidence prediction. Please upload a clearer image."
    )
    st.write(f"### Confidence: {confidence_score:.2f}%")
    st.progress(confidence_score / 100)
    # ============================================
    # Disease Information
    # ============================================

    st.subheader("Disease Information")

    st.write(
    disease_info[predicted_class]
    )

   
    

    # Display Top-3 Predictions
    st.subheader("Top 3 Predictions")

    top_probs, top_classes = torch.topk(
        probabilities,
        k=3
    )

    top_predictions = []

    for i in range(3):

     top_predictions.append({

        "Disease": class_names[
            top_classes[0][i].item()
        ],

        "Confidence (%)": round(
            top_probs[0][i].item()*100,
            2
        )

    })

    st.table(pd.DataFrame(top_predictions))
    # ============================================
# Probability Chart
# ============================================

    chart_data = pd.DataFrame({

    "Disease": class_names,

    "Probability": probabilities.squeeze().numpy()

})

    st.subheader("Prediction Probabilities")

    st.bar_chart(
    chart_data,
    x="Disease",
    y="Probability"
)