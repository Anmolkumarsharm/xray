import streamlit as slt
import torch 
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

slt.title("Chest X-Ray Pneumonia Detection App")
slt.write("Upload your chest image and find if it is NORMAL or PNEUMONIA.")

# Loading Model 
@slt.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("xray_resnet18_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Model load karein
try:
    model = load_model()
except Exception as e:
    slt.warning("Model file load karne mein error hai.")

# Features for uploading image 
uploaded_file = slt.file_uploader("X-Ray Image Choose Karein (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    slt.image(image, caption='Uploaded X-Ray Image', use_container_width=True)

    # In this we are processing the image 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Now we show the result of prediction 
    if slt.button('Predict Result'):
        with slt.spinner("AI model scanning.........."):
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)

            classes = ['Pneumonia', 'normal']
            result = classes[predicted.item()]

            if result == "Pneumonia":
                slt.error(f"Result: **{result}** detected. Meet your doctor as soon as possible.")
            else:
                slt.success(f"Result: **{result}** (Nothing found)")