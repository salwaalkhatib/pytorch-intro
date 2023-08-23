import os
import torch # PyTorch package
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural networks
import torch.optim as optim # optimzer
from dataset.mydataset import MyDataset
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # check if NVIDIA device is visible to torch

model = MODELS["vgg16"].to(DEVICE) # sending model to device
learning_rate = 0.001
epochs = 5 # what is an epoch?
batch_size = 128 
criterion = nn.CrossEntropyLoss() # why this loss function?: https://www.analyticsvidhya.com/blog/2022/08/basic-introduction-to-loss-functions/
optimizer = optim.Adam(model.parameters(), lr = learning_rate) # what's an optimizer?

# number of parameters in the model
print("[INFO] number of parameters in the model: {}".format(sum(p.numel() for p in model.parameters())))

# Create transformations to apply to each image
train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally: More on data augmentation
    transforms.Resize((224, 224)), # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization, why?
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization, why?
])

# Create datasets for training and validation
train_dataset = MyDataset(data_dir='chest_xray/train', transform=train_transform)
val_dataset = MyDataset(data_dir='chest_xray/val', transform=val_transform)
test_dataset = MyDataset(data_dir='chest_xray/test', transform=val_transform)

# Create data loaders for batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# if checkpoints directory not there, create it
if not os.path.exists("checkpoints"):
	os.makedirs("checkpoints")
	print("[INFO] created checkpoints directory")

best_val_acc = 0 # to keep track of best validation accuracy

for epoch in range(epochs):
    # run training loop
    print("[INFO] starting training epoch {}".format(str(epoch+1)))
    loss = train(model, optimizer, criterion, train_loader, DEVICE)
    acc = validate(model, val_loader, DEVICE)
    print(f"[INFO] Epoch {epoch+1}/{epochs}, train loss: {loss:.4f}, val accuracy: {acc:.4f}")
    save_checkpoint(model, optimizer, epoch, "checkpoints/last_epoch.pth") # save checkpoint after each epoch
    if(acc > best_val_acc):
        best_val_acc = acc
        save_checkpoint(model, optimizer, epoch, "checkpoints/best_model.pth", best = True)
