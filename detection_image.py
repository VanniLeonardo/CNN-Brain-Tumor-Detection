import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load the AlexNet model
model = models.alexnet(pretrained=False)
num_classes = 2  # Assuming there are two classes: abnormal and normal
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Map the predicted class index to abnormal or normal
class_names = ['abnormal', 'normal']

# Define the paths to the normal and abnormal image folders
normal_folder = r'YOUR_PATH\Brain_tumor_detection\test\normal'
abnormal_folder = r'YOUR_PATH\Brain_tumor_detection\test\abnormal'

# Load and classify all normal images
normal_images = []
normal_filenames = os.listdir(normal_folder)
for filename in normal_filenames:
    image_path = os.path.join(normal_folder, filename)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]
    normal_images.append((image_path, predicted_class))

# Load and classify all abnormal images
abnormal_images = []
abnormal_filenames = os.listdir(abnormal_folder)
for filename in abnormal_filenames:
    image_path = os.path.join(abnormal_folder, filename)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]
    abnormal_images.append((image_path, predicted_class))

# Calculate accuracy
true_positives = sum([1 for _, predicted_class in abnormal_images if predicted_class == 'abnormal'])
false_negatives = sum([1 for _, predicted_class in abnormal_images if predicted_class == 'normal'])
sensitivity = true_positives / (true_positives + false_negatives)

true_negatives = sum([1 for _, predicted_class in normal_images if predicted_class == 'normal'])
false_positives = sum([1 for _, predicted_class in normal_images if predicted_class == 'abnormal'])
specificity = true_negatives / (true_negatives + false_positives)

total_images = len(normal_images) + len(abnormal_images)
correct_predictions = sum([1 for _, predicted_class in normal_images if predicted_class == 'normal'])
correct_predictions += sum([1 for _, predicted_class in abnormal_images if predicted_class == 'abnormal'])
accuracy = correct_predictions / total_images

print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Accuracy: {accuracy}")

# Plot the results
fig, axs = plt.subplots(2, 5, figsize=(15, 8))
axs = axs.flatten()

# Randomly select images for plotting
random_images = random.sample(normal_images, 5) + random.sample(abnormal_images, 5)

# Plot the randomly selected images
for i, (img_path, predicted_class) in enumerate(random_images):
    axs[i].set_title(f"{predicted_class} (Actual label: {'Normal' if predicted_class == 'normal' else 'Abnormal'})")
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis('off')

plt.tight_layout()
plt.show()
