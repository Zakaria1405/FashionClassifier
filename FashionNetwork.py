import torch
import torch.nn as nn
import torch.nn.functional as F

# class CFashionCNN(nn.Module):
#     def __init__(self, num_classes_clothes, num_classes_gender):
#         super(CFashionCNN, self).__init__()
#         # Input: [batch_size, 1, 256, 256]
#         self.conv1 = nn.Conv2d(1, 32, 3)  # Output: [batch_size, 32, 254, 254]
#         self.conv2 = nn.Conv2d(32, 64, 3) # Output: [batch_size, 64, 252, 252]
        
#         # Adjust pooling layers to accommodate larger input size
#         self.pool = nn.MaxPool2d(2, 2)  # Kernel size 2, stride 2
        
#         # Calculate the input size to the fully connected layer
#         # after applying convolutions and pooling
#         self.fc_input_size = 64 * 62 * 62  # Adjusted for the output size after conv2 and pooling

#         # Fully connected layers for clothes and gender classification
#         self.fc1 = nn.Linear(self.fc_input_size, 128)
#         self.fc_clothes = nn.Linear(128, num_classes_clothes)
#         self.fc_gender = nn.Linear(128, num_classes_gender)

#     def forward(self, x):
#         # Input: [batch_size, 1, 256, 256]
#         x = F.relu(self.conv1(x))
#         # Shape: [batch_size, 32, 254, 254]
#         x = self.pool(x)
#         # Shape: [batch_size, 32, 127, 127]
        
#         x = F.relu(self.conv2(x))
#         # Shape: [batch_size, 64, 252, 252]
#         x = self.pool(x)
#         # Shape: [batch_size, 64, 126, 126]
#         # Flatten the tensor before passing to fully connected layers
#         x = x.view(-1, self.fc_input_size)  # Flattening
        
#         x = F.relu(self.fc1(x))
        
#         # Separate branches for clothes and gender
#         clothes_output = self.fc_clothes(x)
#         gender_output = self.fc_gender(x)

#         return clothes_output, gender_output


import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


class CFashionCNN(nn.Module):
    def __init__(self, num_classes_clothes, num_classes_gender):
        super(CFashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input channels changed to 3, padding added
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.fc_input_size = 128 * 32 * 32  # Adjusted for the output size after conv3 and pooling
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc_clothes = nn.Linear(128, num_classes_clothes)
        self.fc_gender = nn.Linear(128, num_classes_gender)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.contiguous().view(-1, self.fc_input_size)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        clothes_output = self.fc_clothes(x)
        gender_output = self.fc_gender(x)
        return clothes_output, gender_output
