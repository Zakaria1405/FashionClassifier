import pandas as pd
import os
import shutil
import random

# List of desired article types
desired_article_types = [
            'Shirts',
            'Jeans',
            'Tshirts',
            'Socks',
            'Tops',
            'Sweatshirts',
            'Shorts',
            'Dresses'
            # 'Night suits',
            # 'Skirts',
            # 'Blazers',
            # 'Capris',
            # 'Tunics',
            # 'Jackets',
            # 'Lounge Pants',
            # 'Tracksuits',
            # 'Swimwear',
            # 'Sweaters',
            # 'Nightdress',
            # 'Leggings',
            # 'Kurtis',
            # 'Jumpsuit',
            # 'Tights',
            # 'Jeggings',
            # 'Rompers',
            # 'Casual Shoes',
            # 'Formal Shoes',
            # 'Sports Shoes',
            # 'Sandals',
            # 'Heels',
            # 'Flip Flops',
            # 'Booties'
]

# Load the CSV file into a DataFrame
df = pd.read_csv('./Dataset/fashion-dataset/styles.csv', delimiter=';')

# Filter the DataFrame to include only rows with the desired article types and genders 'Men' and 'Women'
filtered_df = df[(df['articletype'].isin(desired_article_types)) & (df['gender'].isin(['Men', 'Women']))]

# Print the number of samples
num_samples = len(filtered_df)
print(f"Number of samples: {num_samples}")

# Define proportions for each split (train, test, val)
train_proportion = 0.9
test_proportion = 0.05
val_proportion = 0.05

# Calculate number of samples for each split
num_train = int(train_proportion * num_samples)
num_test = int(test_proportion * num_samples)
num_val = num_samples - num_train - num_test

# Randomly select samples for each split
random.seed(42)  # For reproducibility
selected_samples_train = filtered_df.sample(n=num_train, random_state=42)
remaining_samples = filtered_df.drop(selected_samples_train.index)
selected_samples_test = remaining_samples.sample(n=num_test, random_state=42)
selected_samples_val = remaining_samples.drop(selected_samples_test.index)

# Define paths for image directories and CSV file
image_dir = './Dataset/fashion-dataset/images'
train_dir = './data/Train/Images'
test_dir = './data/Test/Images'
val_dir = './data/Val/Images'
csv_file = './data/Fashion.csv'

# Create directories if they do not exist
for directory in [train_dir, test_dir, val_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# List to store rows for the CSV file
csv_data = []

# Function to copy images and update CSV data
def copy_images_to_dir(samples_df, dest_directory, split_type):
    for idx, row in samples_df.iterrows():
        image_id = row['id']
        gender = row['gender']
        articletype = row['articletype']
        image_path = os.path.join(image_dir, str(image_id) + ".jpg")
        
        if os.path.exists(image_path):
            # Copy image to destination directory
            img_name = f"img{idx + 1}_{split_type}.jpg"  # Example: img1_train.jpg
            dest_path = os.path.join(dest_directory, img_name)
            shutil.copyfile(image_path, dest_path)
            
            # Append row to CSV data
            csv_data.append([dest_path, split_type, gender, articletype])
            
            print(f"Copied {image_id} to {dest_path}")
        else:
            print(f"Image {image_id} not found at {image_path}")

# Copy images to train, test, and val directories
copy_images_to_dir(selected_samples_train, train_dir, 'train')
copy_images_to_dir(selected_samples_test, test_dir, 'test')
copy_images_to_dir(selected_samples_val, val_dir, 'val')

# Write CSV data to Fashion.csv file
df_csv = pd.DataFrame(csv_data, columns=['Path', 'Split', 'Gender', 'ArticleType'])
df_csv.to_csv(csv_file, index=False)

print(f"CSV file saved to: {csv_file}")
