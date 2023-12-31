import torch
from tqdm import tqdm
from torchvision import models

MODELS = {
	"vgg16": models.vgg16(weights='VGG16_Weights.DEFAULT'),
	"vgg19": models.vgg19(weights='VGG19_Weights.DEFAULT'),
	"resnet": models.resnet50(weights='ResNet50_Weights.DEFAULT'),
	"mobilenet": models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT'),
}

def save_checkpoint(model, optimizer, epoch, path, best = False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # why save optimizer state dictionary?
    }
    torch.save(checkpoint, path)
    if(best):
        print("[INFO] best checkpoint saved at epoch {}".format(epoch))
    else:
        print("[INFO] checkpoint saved at epoch {}".format(epoch))
    
def load_checkpoint(model, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    print("[INFO] checkpoint loaded")
    return model

def train(model, optimizer, criterion, train_loader, DEVICE):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # move data to device
        
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = # TODO: FEED INPUTS TO THE MODEL
        loss = criterion(, ) # TODO: WHAT DO YOU FEED?
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, DEVICE):
    # TODO: Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total * 100