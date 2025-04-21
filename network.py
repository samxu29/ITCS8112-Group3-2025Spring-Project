import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2
from PIL import Image

def initialize_model(num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.features[0]= nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2))
    model_ft.classifier[0] = nn.Dropout(p=0.6, inplace=True)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes
    input_size = 224

    return model_ft, input_size

def transform_img(img_array, input_size=224):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    my_transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
    my_img = my_transform(img)
    my_img = my_img.unsqueeze(0)
    return my_img
        
def predict(model, img_array):
    tensor = transform_img(img_array)
    outputs = model(tensor)
    _, pred = torch.max(outputs,1)
    return pred.item()

num_classes = 29
PATH = './data/param/defined/0.002/Resnet_.pth'  # Updated path to the defined model with correct filename

model, input_size = initialize_model(num_classes)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
