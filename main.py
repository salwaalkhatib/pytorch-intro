import numpy as np # for transformation
import torch # PyTorch package
import torchvision.transforms as transforms # transform data
import torchvision.models as models # load models
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from dataset.mydataset import MyDataset
from tqdm import tqdm # progress bar

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True),
	"mobilenet": models.mobilenet_v2(pretrained=True),
}

model = MODELS["mobilenet"].to(DEVICE)
learning_rate = 0.001
epochs = 10
batch_size = 64
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Create transformations to apply to each image
train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.Resize((128, 128)), # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets for training and validation
train_dataset = MyDataset(data_dir='chest_xray/train', transform=train_transform)
val_dataset = MyDataset(data_dir='chest_xray/val', transform=val_transform)

# Create data loaders for batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(epochs):
	model.train()  # Set the model to training mode
	running_loss = 0.0
	with tqdm(train_loader, unit="batch") as t:
		for inputs, labels in t:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # move data to device
			
			optimizer.zero_grad()  # Zero the gradients
			
			# Forward pass
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			
			# Backpropagation and optimization
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()
			# Update the tqdm progress bar description with the current loss
			t.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
			
		# Validation accuracy
		model.eval()  # Set the model to evaluation mode
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in test_loader:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		validation_accuracy = 100 * correct / total
		print(f'Validation Accuracy: {validation_accuracy:.2f}%')

    

    # Calculate average loss for the epoch
	epoch_loss = running_loss / len(train_loader)

	print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

print('Finished Training!')

# Evaluate the model on the test set


