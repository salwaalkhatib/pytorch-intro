import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Initialize the dataset with the directory containing the data and optional transformations
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.files = []
        for cls in self.classes:
            for img in os.listdir(os.path.join(data_dir, cls)):
                self.files.append(os.path.join(data_dir, cls, img))

    def __len__(self):
        # Return the total number of images in the dataset split
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        # Load the image using PIL (Pillow) library
        image = Image.open(img_path).convert("RGB")  # Convert to RGB format
        # Apply optional transformations to the image
        if self.transform:
            image = self.transform(image)
        # Encode the label based on the class (0 for 'normal', 1 for 'pneumonia')
        label = 0 if 'NORMAL' in img_path else 1
        return image, label
