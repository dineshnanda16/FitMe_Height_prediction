#all data with traing also
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Read CSV file containing image names and heights
csv_file = './Output_data.csv'
data = pd.read_csv(csv_file)

# Function to extract height in inches and convert to centimeters
def extract_height_in_cms(height_weight_str):
    # Regular expression to match heights like 4' 10"
    match = re.match(r"\s*(\d+)' (\d+)\"", height_weight_str)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        total_inches = feet * 12 + inches  # Convert to total inches
        return total_inches * 2.54  # Convert inches to centimeters
    return None

# Apply the function to the 'Height & Weight' column to create 'Height_in_cms'
data['Height_in_cms'] = data['Height & Weight'].apply(extract_height_in_cms)

# Ensure 'Height_in_cms' was created successfully
if 'Height_in_cms' not in data.columns:
    print("Height_in_cms column was not created!")
else:
    print("Height_in_cms column created successfully.")

# Assuming 'Filename' is the column for image filenames
image_names = data['Filename'].values
heights = data['Height_in_cms'].values
image_folder = './images'  # Define the image folder path

# Define a custom dataset
class HeightDataset(Dataset):
    def __init__(self, image_folder, image_names, heights, transform=None):
        self.image_folder = image_folder
        self.image_names = image_names
        self.heights = heights.astype(float)  # Ensure heights are of type float
        self.transform = transform

        # Filter out invalid images and heights
        valid_indices = []
        for idx in range(len(image_names)):
            image_path = os.path.join(image_folder, image_names[idx])
            try:
                Image.open(image_path).convert('RGB')  # Check if image can be opened
                valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping invalid image: {image_path}. Error: {e}")

        # Keep only valid images and corresponding heights
        self.image_names = self.image_names[valid_indices]
        self.heights = self.heights[valid_indices]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Construct full image path
        image_path = os.path.join(self.image_folder, self.image_names[idx])
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
        height = self.heights[idx]

        # Apply any image transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(height, dtype=torch.float32)  # Ensure height is a Float tensor

# Define the transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),            # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize between -1 and 1
])

# Create Dataset
dataset = HeightDataset(image_folder=image_folder, image_names=image_names, heights=heights, transform=transform)

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the CNN model for height prediction
class HeightPredictionCNN(nn.Module):
    def __init__(self):
        super(HeightPredictionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Assuming input size is 128x128
        self.fc2 = nn.Linear(128, 1)  # Output size is 1 (height prediction)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = HeightPredictionCNN()

# Define the loss function (Mean Squared Error) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10  # Adjust based on your dataset size
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch in train_loader:
        images, heights = batch
        images, heights = images.to(device), heights.to(device)  # Move to device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.view(-1), heights)  # Adjust for shape

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete.")

# Save the model
torch.save(model.state_dict(), 'height_prediction_model.pth')

# Function to predict height for new images
def predict_height(image_path, model, transform):
    model.eval()  # Set model to evaluation mode
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
            return output.item()  # Return the predicted height in cm
    except Exception as e:
        print(f"Error predicting height for {image_path}: {e}")
        return None

# Example usage for prediction
test_image_path = './test_img/dad.jpg'  # Change this to your test image path
predicted_height = predict_height(test_image_path, model, transform)
print(f"Predicted Height: {predicted_height} cm")

# Visualize results
# Collect actual heights and predicted heights for visualization (you can use the same DataLoader)
actual_heights = []
predicted_heights = []

# Evaluate and collect predictions
with torch.no_grad():
    for batch in train_loader:
        images, heights = batch
        images = images.to(device)

        outputs = model(images)
        actual_heights.extend(heights.view(-1).tolist())
        predicted_heights.extend(outputs.view(-1).tolist())

# Scatter plot for actual vs predicted heights
plt.scatter(actual_heights, predicted_heights)
plt.xlabel('Actual Heights (cm)')
plt.ylabel('Predicted Heights (cm)')
plt.title('Actual vs Predicted Heights')
plt.plot([min(actual_heights), max(actual_heights)], [min(actual_heights), max(actual_heights)], color='red')  # Diagonal line
plt.show()
