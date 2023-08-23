from utils import load_checkpoint, validate, MODELS
from dataset.mydataset import MyDataset
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
val_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = MyDataset(data_dir='chest_xray/test', transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# load best checkpoint
model = MODELS["vgg16"].to(DEVICE)
model = load_checkpoint(model, "checkpoints/best_model.pth")

# inference
acc = validate(model, test_loader, DEVICE)
print("[INFO] Accuracy: {:.4f}".format(acc))


