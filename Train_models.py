#!/usr/bin/env python
# coding: utf-8
# libraries
import shutil
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import torchsummary
import sklearn
import splitfolders
from sklearn import metrics
import seaborn as sn
from tqdm.notebook import tqdm, trange
from tqdm.notebook import tqdm
# Define the paths for train set and the test set
path_ASL_alphabet_train = "./data/asl_alphabet_train/asl_alphabet_train/"
path_ASL_alphabet_test = "./data/asl_alphabet_test/asl_alphabet_test/"
input_path = path_ASL_alphabet_train
sample_path = './data/output/'
data_dir = './data/grey/'
create_gray = False
model_name = "squeezenet"  # Models to choose from [resnet, defined, squeezenet]
num_classes = 29  # Number of classes in the dataset
batch_size = 100  # Batch size for training
num_epochs = 40  # Number of epochs to train for
feature_extract = False  # Flag for feature extracting.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.02
need_to_split = False


def hist_normalization(img, a=0, b=255):
    c = img.min()
    d = img.max()
    out = img.copy()
    out = (b-a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b
    out = out.astype(np.uint8)
    return out


def train_model(model, dataloaders, criterion, optimizer, path, num_epochs, is_inception=False):
    since = time.time()

    PATH = path

    epochs = []

    val_acc_history = []
    val_loss_history = []

    train_acc_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_count = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2

                    else:

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_count += 1  # Since we started at 0
                if running_count % 100 ==0:
                    print(running_count)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model  #This is taking the model with the best epoch weights.
            # does this avoid over-training???
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
                epochs.append(epoch)
            elif phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epochs, train_loss_history, train_acc_history, val_loss_history, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "defined":
        """ our model
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.features[0]= nn.Conv2d(3, 96, kernel_size=(5, 5), stride=(2, 2))
        model_ft.classifier[0] = nn.Dropout(p=0.6, inplace=True)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def show_train_plots(x, train_loss, train_acc, val_loss, val_acc):
    plt.figure()
    plt.plot( x,[t.cpu().numpy() for t in train_acc], 'r.--', label = 'training accuracy')
    plt.plot( x,[t.cpu().numpy() for t in val_acc], 'bo', label = 'validation accuracy')
    plt.title('training and val accuracy')
    plt.legend()
    plt.savefig('./training_and_val_accuracy.png')

    plt.figure()
    plt.plot( x,train_loss, 'bo', label = 'training loss')
    plt.plot (x, val_loss, 'b', label = 'validation loss', color='red')
    plt.title ('training and val loss')
    plt.legend()
    plt.savefig('training and val loss.png')


if __name__ == '__main__':

    if not(os.path.isdir(sample_path)): #if sample path not created, make it.
        os.mkdir(sample_path)


    if create_gray:
        for folder_name in os.listdir(input_path): #populate sample with with images
            mk_dir_path = os.path.join(sample_path,folder_name)

            if not(os.path.isdir(mk_dir_path)):
                os.mkdir(mk_dir_path)

            for image_name in tqdm(os.listdir(input_path+ folder_name)):
                img = cv2.imread(os.path.join(input_path,folder_name,image_name) )
                # convert images to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Normalize grayscale pixels to 0 ~ 255
                normal_gray_img = hist_normalization(gray_img)

                cv2.imwrite(os.path.join(mk_dir_path,image_name), normal_gray_img)

    if need_to_split:
        tqdm(splitfolders.ratio(input=sample_path, output=data_dir, seed=1337, ratio=(0.8, 0.1, 0.1)))

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)

    # Print the model we just instantiated
    print(model_ft)

    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    print('Device: ', device)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=learning_rate)

    # Set up the loss fxn
    criterion = nn.CrossEntropyLoss()
    PATH = "./Resnet.pth"
    # Train and evaluate
    model_ft, epoch, train_loss, train_acc, val_loss, val_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, PATH, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    torch.save(model_ft.state_dict(), PATH)

    show_train_plots (epoch, train_loss,train_acc,val_loss, val_acc)

    print(train_acc)
    print(val_acc)
