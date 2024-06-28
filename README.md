# Fashion Project Documentation

## Installation

1. **Create an Environment**:
    ```bash
    conda create --name Fashion python=3.9
    conda activate Fashion
    ```

2. **Install Repository Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

1. **Download the Dataset**:
    - The dataset for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

2. **Set Up Directory Structure**:
    After downloading, organize your directories as follows:
    ```
    Fashion/
    ├── data/
    │   ├── test/
    │   ├── train/
    │   ├── Fashion/
    ├── Dataset/
    │   └── fashion-dataset/
    │       ├── fashion-dataset/
    │       ├── images/
    │       ├── styles/
    │       ├── images.csv
    │       └── styles.csv
    ```

3. **Run the Preprocessor**:
    - From inside the repository, execute the following command to run the preprocessor:
      ```bash
      python Preprocessor.py
      ```
    - The preprocessor will organize the dataset in a format suitable for the main network.

## Active Article Types

The model is trained to distinguish between the following items:
- 'Shirts': 1
- 'Jeans': 2
- 'Tshirts': 3
- 'Socks': 4
- 'Tops': 5
- 'Sweatshirts': 6
- 'Shorts': 7
- 'Dresses': 8

You can add more items and retrain the model as needed.

## Training the Model

1. **Configure Training Settings**:
    - Review and adjust the configuration settings in the file `config/CFashion_config.yaml`.

2. **Execute the Training**:
    ```bash
    python main.py
    ```
