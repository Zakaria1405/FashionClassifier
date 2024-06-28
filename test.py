import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from FashionNetwork import CFashionCNN

# Load the trained model
def load_model(weights_path, device, num_classes_gender, num_classes_clothes):
    model = CFashionCNN(num_classes_clothes=num_classes_clothes, num_classes_gender=num_classes_gender).to(device)
    if os.path.exists(weights_path):
        print("Loading saved model weights...")
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# Function to predict gender and article type from an image
def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        article_outputs, gender_outputs = model(image)
        _, gender_predicted = torch.max(gender_outputs, 1)
        _, article_predicted = torch.max(article_outputs, 1)
    
    predicted_gender = gender_classes[gender_predicted.item()]
    predicted_article = article_classes[article_predicted.item()]
    
    return predicted_gender, predicted_article

# Paths and parameters
weights_path = './Fashion.pth'
image_path = './Model6.jpeg'  # Replace with the path to the image you want to predict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class labels for the Fashion dataset
gender_classes = ['Women', 'Men']
article_classes = [
    'Shirts', 'Jeans', 'Tshirts', 'Socks', 'Tops', 'Sweatshirts', 'Shorts', 'Dresses'
]

# Transform for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Main function to run the prediction
def main():
    num_classes_gender = len(gender_classes)
    num_classes_clothes = len(article_classes)
    
    # Load the model
    model = load_model(weights_path, device, num_classes_gender, num_classes_clothes)
    
    # Predict the gender and article type for the input image
    predicted_gender, predicted_article = predict_image(model, image_path, transform, device)
    
    print(f"Predicted Gender: {predicted_gender}")
    print(f"Predicted Article: {predicted_article}")

if __name__ == "__main__":
    main()
