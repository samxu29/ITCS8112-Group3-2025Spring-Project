import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os
from PIL import Image
import shutil

model_name = "defined" # resnet|defined|squeezenet
lr = "0.02" # 0.02|0.002|0.0002

path_ASL_alphabet_test = "./data/grey/test/"
model_path = "./data/param/" + model_name + "/" + lr + "/Resnet.pth"
input_size = 224

def initialize_model(model_name):
    # Initialize the model for testing based on the model name
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 29)
    elif model_name == "defined":
        model_ft = models.squeezenet1_0(pretrained=True)
        model_ft.features[0]= nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2))
        model_ft.classifier[0] = nn.Dropout(p=0.6, inplace=True)
        model_ft.classifier[1] = nn.Conv2d(512, 29, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = 29

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_0(pretrained=True)
        model_ft.classifier[1] = nn.Conv2d(512, 29, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = 29

    return model_ft

model_ft = initialize_model(model_name)
model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# Use the loaded model for testing
model_ft.eval()


# Define the data transformations for the test set
test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Create a dataset and a dataloader for the grey test set
test_dataset = datasets.ImageFolder(path_ASL_alphabet_test, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# print("test_dataset.classes", test_dataset.classes)
# print("test_dataset.class_to_idx", test_dataset.class_to_idx)
# print("test_dataset.imgs", test_dataset.imgs)

# Set up the lists for true labels and predicted labels
y_true = []
y_pred = []

# Iterate over the test set and make predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to("cpu")
        labels = labels.to("cpu")
        # print("inputs", inputs)
        # print("labels", labels)
        outputs = model_ft(inputs)
        # print("outputs", outputs)
        _, preds = torch.max(outputs, 1)
        # print("preds", preds)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute the accuracy of the model
accuracy = accuracy_score(y_true, y_pred)
print("Model: %s, Learning Rate= %s, Accuracy on the grey test set: %.2f%%" % (model_name, lr, accuracy * 100))

# Create a dataset and a dataloader for the Kaggle test set
Kaggle_test_path = "./data/asl_alphabet_test/asl_alphabet_test/"

# Create a dataset and a dataloader for the grey test set
test_dataset = datasets.ImageFolder(Kaggle_test_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Set up the lists for true labels and predicted labels
y_true = []
y_pred = []

# Iterate over the test set and make predictions
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to("cpu")
        labels = labels.to("cpu")
        # print("inputs", inputs)
        # print("labels", labels)
        outputs = model_ft(inputs)
        # print("outputs", outputs)
        _, preds = torch.max(outputs, 1)
        # print("preds", preds)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute the accuracy of the model
accuracy = accuracy_score(y_true, y_pred)
print("Model: %s, Learning Rate= %s, Accuracy on the Kaggle test set: %.2f%%" % (model_name, lr, accuracy * 100))

