import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize the dataset with the directory containing the data and optional transformations
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.files = os.listdir(os.path.join(data_dir, self.classes[0])) + os.listdir(os.path.join(data_dir, self.classes[1]))

    def __len__(self):
        # Return the total number of images in the dataset
        return sum(len(os.listdir(os.path.join(self.data_dir, cls))) for cls in self.classes)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        cls ='PNEUMONIA' if 'person' in img_name else 'NORMAL'
        img_path = os.path.join(self.data_dir, cls, img_name)
        # Load the image using PIL (Pillow) library
        image = Image.open(img_path).convert("RGB")  # Convert to RGB format
        # Apply optional transformations to the image
        if self.transform:
            image = self.transform(image)
        # Encode the label based on the class (0 for 'normal', 1 for 'pneumonia')
        label = 0 if cls == 'NORMAL' else 1
        return image, label
