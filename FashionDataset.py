import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CFashionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        """
        Args:
            root_dir (string): Path to the CSV file
            split (string): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Check if the split is valid
        if self.split not in ["train", "val", "test"]:
            raise ValueError("Split must be one of 'train', 'val', 'test'")
        
        # Load csv file
        df = pd.read_csv(root_dir, delimiter=';')

        # Filter the DataFrame to include only rows with desired Split
        filtered_df = df[df['Split'] == self.split]

        # Mapping ArticleType to numbers
        article_type_mapping = {
            'Shirts': 0,
            'Jeans': 1,
            'Tshirts': 2,
            'Socks': 3,
            'Tops': 4,
            'Sweatshirts': 5,
            'Shorts': 6,
            'Dresses': 7}
            # 'Night suits': 9,
            # 'Skirts': 10,
            # 'Blazers': 11,
            # 'Capris': 12,
            # 'Tunics': 13,
            # 'Jackets': 14,
            # 'Lounge Pants': 15,
            # 'Tracksuits': 16,
            # 'Swimwear': 17,
            # 'Sweaters': 18,
            # 'Nightdress': 19,
            # 'Leggings': 20,
            # 'Kurtis': 21,
            # 'Jumpsuit': 22,
            # 'Tights': 23,
            # 'Jeggings': 24,
            # 'Rompers': 25,
            # 'Casual Shoes': 26,
            # 'Formal Shoes': 27,
            # 'Sports Shoes': 28,
            # 'Sandals': 29,
            # 'Heels': 30,
            # 'Flip Flops': 31,
            # 'Booties': 32
        #}

        # Map gender to numbers
        gender_mapping = {'Men': 1, 'Women': 0}

        self.image_paths = filtered_df['Path'].tolist()
        self.ArticleTypes = [article_type_mapping[atype] for atype in filtered_df['ArticleType'].tolist()]
        self.genders = [gender_mapping[gender] for gender in filtered_df['Gender'].tolist()]
        self.num_samples = len(self.image_paths)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # convert image to grayscale
        articletype = self.ArticleTypes[idx]
        gender = self.genders[idx]
        

        if self.transform:
            image = self.transform(image)

        return img_path, image, (articletype, gender)
