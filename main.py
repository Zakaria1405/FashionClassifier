# import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import os
# from tqdm import tqdm  

# from FashionDataset import CFashionDataset
# from FashionNetwork import CFashionCNN
# from helper import *

# # For reproducibility
# torch.manual_seed(42)
# np.random.seed(42)

# # Setting up device for GPU usage if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on {device}.")

# # Path to save/load model weights
# weights_path = './Fashion.pth'

# # Convert image to Pytorch Tensor Format
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()
# ])

# # Load configuration
# config = load_config('./config/CFashion_config.yaml')

# # Initialize datasets
# trainset = CFashionDataset(root_dir="./data/Fashion.csv", split="train", transform=transform)
# testset = CFashionDataset(root_dir="./data/Fashion.csv", split="test", transform=transform)
# valset = CFashionDataset(root_dir="./data/Fashion.csv", split="val", transform=transform)

# # Initialize data loaders
# trainloader = torch.utils.data.DataLoader(trainset,
#                                           batch_size=config['data_loaders']['train']['batch_size'],
#                                           shuffle=config['data_loaders']['train']['shuffle'],
#                                           num_workers=config['data_loaders']['train']['num_workers'])

# testloader = torch.utils.data.DataLoader(testset,
#                                          batch_size=config['data_loaders']['test']['batch_size'],
#                                          shuffle=config['data_loaders']['test']['shuffle'],
#                                          num_workers=config['data_loaders']['test']['num_workers'])

# valloader = torch.utils.data.DataLoader(valset,
#                                         batch_size=config['data_loaders']['val']['batch_size'],
#                                         shuffle=config['data_loaders']['val']['shuffle'],
#                                         num_workers=config['data_loaders']['val']['num_workers'])


# # Define the class labels for the Fashion dataset
# gender_classes = ['Women', 'Men']
# article_classes = [
#             'Shirts',
#             'Jeans',
#             'Tshirts',
#             'Socks',
#             'Tops',
#             'Sweatshirts',
#             'Shorts',
#             'Dresses'
#             # 'Night suits',
#             # 'Skirts',
#             # 'Blazers',
#             # 'Capris',
#             # 'Tunics',
#             # 'Jackets',
#             # 'Lounge Pants',
#             # 'Tracksuits',
#             # 'Swimwear',
#             # 'Sweaters',
#             # 'Nightdress',
#             # 'Leggings',
#             # 'Kurtis',
#             # 'Jumpsuit',
#             # 'Tights',
#             # 'Jeggings',
#             # 'Rompers',
#             # 'Casual Shoes',
#             # 'Formal Shoes',
#             # 'Sports Shoes',
#             # 'Sandals',
#             # 'Heels',
#             # 'Flip Flops',
#             # 'Booties'
# ]

# # Map article types to numeric labels
# article_type_mapping = {
#         'Shirts': 1,
#         'Jeans': 2,
#         'Tshirts': 3,
#         'Socks': 4,
#         'Tops': 5,
#         'Sweatshirts': 6,
#         'Shorts': 7,
#         'Dresses': 8
#         # 'Night suits': 9,
#         # 'Skirts': 10,
#         # 'Blazers': 11,
#         # 'Capris': 12,
#         # 'Tunics': 13,
#         # 'Jackets': 14,
#         # 'Lounge Pants': 15,
#         # 'Tracksuits': 16,
#         # 'Swimwear': 17,
#         # 'Sweaters': 18,
#         # 'Nightdress': 19,
#         # 'Leggings': 20,
#         # 'Kurtis': 21,
#         # 'Jumpsuit': 22,
#         # 'Tights': 23,
#         # 'Jeggings': 24,
#         # 'Rompers': 25,
#         # 'Casual Shoes': 26,
#         # 'Formal Shoes': 27,
#         # 'Sports Shoes': 28,
#         # 'Sandals': 29,
#         # 'Heels': 30,
#         # 'Flip Flops': 31,
#         # 'Booties': 32
# }

# # Map gender to numeric labels
# gender_mapping = {'Women': 0, 'Men': 1}


# def evaluate_model(model, valloader):
#     """
#     Evaluate a PyTorch model on a validation set using classification report and confusion matrix.
    
#     Args:
#     - model (torch.nn.Module): The trained PyTorch model to evaluate
#     - valloader (torch.utils.data.DataLoader): DataLoader for the validation set
   
#     Returns:
#     - gender_classification_report_str (str): String representation of the gender classification report
#     - article_classification_report_str (str): String representation of the article type classification report
#     - gender_confusion_matrix_plot (matplotlib.figure.Figure): Matplotlib figure object of the gender confusion matrix plot
#     - article_confusion_matrix_plot (matplotlib.figure.Figure): Matplotlib figure object of the article type confusion matrix plot
#     - image_predictions (list): List of dictionaries containing image paths, predicted gender, and predicted article type
#     """
#     # Set the model to evaluation mode
#     model.eval()
    
#     # Lists to store all predictions and true labels
#     all_gender_preds = []
#     all_gender_labels = []
#     all_article_preds = []
#     all_article_labels = []
    
#     # List to store predictions for each image
#     image_predictions = []
    
#     # We don't want to compute gradients during evaluation
#     with torch.no_grad():
#         # Iterate over all batches in the validation loader
#         for image_path, images, (gender_labels, article_labels) in valloader:
#             # Transfer images and labels to the computational device
#             images = images.to(device)
#             gender_labels = gender_labels.to(device)
#             article_labels = article_labels.to(device)
            
#             # Pass the images through the model to get predictions
#             article_outputs, gender_outputs  = model(images)
            
#             # Get the predicted class indices for gender and article type
#             _, gender_predicted = torch.max(gender_outputs, 1)
#             _, article_predicted = torch.max(article_outputs, 1)
            
#             # Convert predicted indices to actual class names
#             gender_pred_labels = []
#             for label_g in gender_predicted.cpu().numpy():
#                 if label_g < len(gender_classes):
#                     gender_pred_labels.append(gender_classes[label_g])
#                 else:
#                     gender_pred_labels.append('Unknown')
#                     print(f"Warning: Predicted label {label_g} out of range of gender_classes")
            
#             article_pred_labels = []
#             for label_a in article_predicted.cpu().numpy():
#                 if label_a < len(article_classes):
#                     article_pred_labels.append(article_classes[label_a])
#                 else:
#                     article_pred_labels.append('Unknown')
#                     print(f"Warning: Predicted label {label_a} out of range of article_classes")
            
#             # Extend the lists with predictions from this batch
#             all_gender_preds.extend(gender_pred_labels)
#             all_gender_labels.extend(gender_classes[label_g])
#             all_article_preds.extend(article_pred_labels)
#             all_article_labels.extend(article_classes[label_a])
            

            
#             # Store predictions for each image in this batch

#             image_predictions.append({
#             'image_path': image_path,
#             'predicted_gender': gender_pred_labels,
#             'predicted_article': article_pred_labels
#             })


#     for prediction in image_predictions:
#         path_S = prediction['image_path']
#         pred_g = prediction['predicted_gender']
#         pred_a = prediction['predicted_article']
        
#         print(f"Path: {path_S}")
#         print(f"Gender: {pred_g}")
#         print(f"Article: {pred_a}")




# def train_model(model):
#     """
#     Train the given model on the training dataset.
    
#     Args:
#     - model (torch.nn.Module): The model to train.
    
#     Returns:
#     - model (torch.nn.Module): The trained model.
#     """
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     num_epochs = config['epochs']

#     epoch_losses = []
    
#     for epoch in range(num_epochs):
#         model.train()
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         progress_bar = tqdm(trainloader, desc='Training', leave=False)
        
#         running_loss = 0.0
#         total_batches = 0
        
#         for batch_idx, (images_path ,images, (article_labels, gender_labels)) in enumerate(progress_bar):
#             images = images.to(device)
#             gender_labels = gender_labels.to(device)
#             article_labels = article_labels.to(device)

#             optimizer.zero_grad()

#             outputs_article, outputs_gender = model(images)
#             loss_article = criterion(outputs_article, article_labels)
#             loss_gender = criterion(outputs_gender, gender_labels)

#             loss = loss_article + loss_gender
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             total_batches += 1

#             progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
#         # Calculate average loss for the epoch
#         epoch_loss = running_loss / total_batches
#         epoch_losses.append(epoch_loss)
#         print(f"Average Loss for Epoch {epoch+1}: {epoch_loss:.4f}")


#     # Plotting the training loss and saving it to a file
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Loss')
#     plt.title('Training Loss over Epochs')
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot
#     plot_path = 'training_loss_plot.png'
#     plt.savefig(plot_path)
#     print(f"Training loss plot saved to {plot_path}")
#     plt.close()

#     return model


# def main():
#     num_classes_gender = len(gender_classes)
#     num_classes_clothes = len(article_classes)
    
#     # Initialize the model
#     model = CFashionCNN(num_classes_clothes=num_classes_clothes, num_classes_gender=num_classes_gender).to(device)
    
#     # Load pretrained weights if available and specified
#     if os.path.exists(weights_path):
#         print("Loading saved model weights...")
#         model.load_state_dict(torch.load(weights_path))
    
#     # Train the model if specified
#     if config['Train']:
#         print("Training model...")
#         model = train_model(model)

#         # Save model weights
#         torch.save(model.state_dict(), weights_path)
#         print(f"Model weights saved to {weights_path}")
    
#     # Evaluate the model on the validation set
#     print("Evaluating model...")
#     evaluate_model(model, valloader)
    
   

# if __name__ == "__main__":
#     main()
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm  
from sklearn.metrics import accuracy_score

from FashionDataset import CFashionDataset
from FashionNetwork import CFashionCNN
from helper import *
import pandas as pd

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Setting up device for GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}.")

# Path to save/load model weights
weights_path = './Fashion.pth'

# Convert image to Pytorch Tensor Format
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load configuration
config = load_config('./config/CFashion_config.yaml')

# Initialize datasets
trainset = CFashionDataset(root_dir="./data/Fashion.csv", split="train", transform=transform)
testset = CFashionDataset(root_dir="./data/Fashion.csv", split="test", transform=transform)
valset = CFashionDataset(root_dir="./data/Fashion.csv", split="val", transform=transform)

# Initialize data loaders
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=config['data_loaders']['train']['batch_size'],
                                          shuffle=config['data_loaders']['train']['shuffle'],
                                          num_workers=config['data_loaders']['train']['num_workers'])

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=config['data_loaders']['test']['batch_size'],
                                         shuffle=config['data_loaders']['test']['shuffle'],
                                         num_workers=config['data_loaders']['test']['num_workers'])

valloader = torch.utils.data.DataLoader(valset,
                                        batch_size=config['data_loaders']['val']['batch_size'],
                                        shuffle=config['data_loaders']['val']['shuffle'],
                                        num_workers=config['data_loaders']['val']['num_workers'])


# Define the class labels for the Fashion dataset
gender_classes = ['Women', 'Men']
article_classes = [
    'Shirts', 'Jeans', 'Tshirts', 'Socks', 'Tops', 'Sweatshirts', 'Shorts', 'Dresses'
]

# Map article types to numeric labels
article_type_mapping = {
    'Shirts': 0, 'Jeans': 1, 'Tshirts': 2, 'Socks': 3,
    'Tops': 4, 'Sweatshirts': 5, 'Shorts': 6, 'Dresses': 7
}

# Map gender to numeric labels
gender_mapping = {'Women': 0, 'Men': 1}


def evaluate_model(model):
    model.eval()
    all_gender_preds = []
    all_gender_labels = []
    all_article_preds = []
    all_article_labels = []
    image_predictions = []
    
    with torch.no_grad():
        for image_path, images, (gender_labels, article_labels) in testloader:
            images = images.to(device)
            gender_labels = gender_labels.to(device)
            article_labels = article_labels.to(device)
            
            article_outputs, gender_outputs  = model(images)
            
            _, gender_predicted = torch.max(gender_outputs, 1)
            _, article_predicted = torch.max(article_outputs, 1)
            
            all_gender_preds.extend(gender_predicted.cpu().numpy())
            all_gender_labels.extend(gender_labels.cpu().numpy())
            all_article_preds.extend(article_predicted.cpu().numpy())
            all_article_labels.extend(article_labels.cpu().numpy())
            
            image_predictions.extend([{
                'image_path': path,
                'predicted_gender': gender_classes[pred_g],
                'predicted_article': article_classes[pred_a]
            } for path, pred_g, pred_a in zip(image_path, gender_predicted.cpu().numpy(), article_predicted.cpu().numpy())])

    return image_predictions

def train_model(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    num_epochs = config['epochs']
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(trainloader, desc='Training', leave=False)
        
        running_loss = 0.0
        total_batches = 0
        
        for batch_idx, (images_path, images, (article_labels, gender_labels)) in enumerate(progress_bar):
            images = images.to(device)
            gender_labels = gender_labels.to(device)
            article_labels = article_labels.to(device)

            optimizer.zero_grad()
            outputs_article, outputs_gender = model(images)
            loss_article = criterion(outputs_article, article_labels)
            loss_gender = criterion(outputs_gender, gender_labels)
            loss = loss_article + loss_gender
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1
            progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})
        
        epoch_loss = running_loss / total_batches
        epoch_losses.append(epoch_loss)
        print(f"Average Loss for Epoch {epoch+1}: {epoch_loss:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = 'training_loss_plot.png'
    plt.savefig(plot_path)
    print(f"Training loss plot saved to {plot_path}")
    plt.close()
    
    return model


def main():
    num_classes_gender = len(gender_classes)
    num_classes_clothes = len(article_classes)
    
    model = CFashionCNN(num_classes_clothes=num_classes_clothes, num_classes_gender=num_classes_gender).to(device)
    
    if os.path.exists(weights_path):
        print("Loading saved model weights...")
        model.load_state_dict(torch.load(weights_path))
    
    if config['Train']:
        print("Training model...")
        model = train_model(model)
        torch.save(model.state_dict(), weights_path)
        print(f"Model weights saved to {weights_path}")
    
    print("Evaluating model...")
    image_predictions = evaluate_model(model)  # Changed to use testloader
    
    # Load the CSV file with true labels
    true_data = pd.read_csv('./data/Fashion.csv', delimiter=';')
    
    # Filter the CSV to only include rows that match the test set
    true_data_filtered = true_data[true_data["Split"] == "test"]
    
    # Create lists to store true labels and predictions
    true_genders = []
    pred_genders = []
    true_articles = []
    pred_articles = []
    
    for prediction in image_predictions:
        # Get the corresponding row in the filtered DataFrame
        true_row = true_data_filtered[true_data_filtered['Path'] == prediction['image_path']]
    
        if not true_row.empty:
            true_gender = true_row['Gender'].values[0]
            predicted_gender = prediction['predicted_gender']
            # Now you can use the full predicted gender label
            print(f"True gender: {true_gender}, Predicted gender: {predicted_gender}")
            
            
            # Append the true and predicted labels to respective lists
            true_genders.append(gender_classes.index(true_gender))
            pred_genders.append(gender_classes.index(predicted_gender))

            true_article = true_row['ArticleType'].values[0]
            predicted_article = prediction['predicted_article']

            # Debugging: Print the true and predicted article types
            print(f"True Article: {true_article}, Predicted Article: {predicted_article}")

            try:
                true_articles.append(article_classes.index(true_article))
                pred_articles.append(article_classes.index(predicted_article))
            except ValueError as e:
                print(f"Error: {e}")
                print(f"Skipping this prediction due to mismatch in article type classes.")

    # Calculate accuracy for gender and article type predictions
    if true_genders and pred_genders:
        gender_accuracy = accuracy_score(true_genders, pred_genders)
        print(f"Gender Prediction Accuracy: {gender_accuracy * 100:.2f}%")
    else:
        print("No valid gender predictions to calculate accuracy.")

    if true_articles and pred_articles:
        article_accuracy = accuracy_score(true_articles, pred_articles)
        print(f"Article Type Prediction Accuracy: {article_accuracy * 100:.2f}%")
    else:
        print("No valid article type predictions to calculate accuracy.")


if __name__ == "__main__":
    main()
