import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io

def main():
    st.title("Height Prediction from Image")
    
    # Sidebar for uploading an image
    st.sidebar.header("Upload your image")
    test_image = st.sidebar.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    # Load pre-trained model (ensure the model path is correct)
    model_path = './height_prediction_model.pth'
    model = HeightPredictionCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    if test_image is not None:
        # Display the uploaded image
        image = Image.open(test_image).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict height for the uploaded image
        predicted_height = predict_height(image, model, transform, device)
        if predicted_height is not None:
            st.write(f"Predicted Height: {predicted_height:.2f} cm")
        else:
            st.write("Error predicting height.")

# Function to predict height for an uploaded image
def predict_height(image, model, transform, device):
    model.eval()
    try:
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            return output.item()  # Return the predicted height in cm
    except Exception as e:
        return None

# CNN Model definition
class HeightPredictionCNN(nn.Module):
    def __init__(self):
        super(HeightPredictionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Run the Streamlit app
if __name__ == '__main__':
    main()
