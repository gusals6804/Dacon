import numpy as np
import pandas as pd
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as data_utils
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split

# For image-keypoints data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Prefix data directory
prefix_dir = '.'

# Top level data directory. Here we assume the format of the directory conforms
# to the ImageFolder structure
train_dir = f'{prefix_dir}/data/train_imgs'

# Models to choose from torchvision
model_name = 'hrnet'
model_ver = 'w48'

# Number of classes in the dataset
num_classes = 48

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs and earlystop to train for
num_epochs = 50

num_splits = 10
num_earlystop = 5

# Iput size for resize imgae
input_w = 1024
input_h = 1024

# Learning rate for optimizer
learning_rate = 0.01

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False

df = pd.read_csv(f'{prefix_dir}/data/train_df.csv')
print(df.head())

error_list=[317, 869, 873, 877, 911, 1559, 1560, 1562, 1566, 1575, 1577, 1578, 1582, 1606, 1607, 1622,
            1623, 1624, 1625, 1629, 3968, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124,
            4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139,
            4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154,
            4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169,
            4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185,
            4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194]

df.drop(error_list, inplace=True)
print(len(df))

imgs = df.iloc[:, 0].to_numpy()
motions = df.iloc[:, 1:]
columns = motions.columns.to_list()[::2]
class_labels = [label.replace('_x', '').replace('_y', '') for label in columns]
keypoints = []
for motion in motions.to_numpy():
    a_keypoints = []
    for i in range(0, motion.shape[0], 2):
        a_keypoints.append((float(motion[i]), float(motion[i+1])))
    keypoints.append(a_keypoints)
keypoints = np.array(keypoints)



def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, earlystop=0, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    earlystop_value = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999999999

    for epoch in range(num_epochs):
        epoch_since = time.time()
        if earlystop and earlystop_value >= earlystop:
            break

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs.float(), labels.float())
                        loss2 = criterion(aux_outputs.float(), labels.float())
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs.float(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    count += 1
                    if count % 10 == 0:
                        print(count)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # for regression
                #running_corrects += torch.sum(outputs == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_time_elapsed = time.time() - epoch_since
            print('{} ({}) Loss: {:.4f} Elapsed time: {:.0f}m {:.0f}s'.format(
                phase, len(dataloaders[phase].dataset), epoch_loss, epoch_time_elapsed // 60,
                                                                               epoch_time_elapsed % 60))
            # neptune.log_metric(f'{phase}_loss', epoch_loss)
            # neptune.log_metric(f'{phase}_acc', epoch_acc)
            if phase == 'val':
                lr_scheduler.step(epoch_loss)

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    earlystop_value = 0
                    best_model = model
                    path = './hrnet_models/'
                    torch.save(best_model.state_dict(), f'{path}hrnet_1024{epoch_loss}.pth')
                else:
                    earlystop_value += 1
                val_loss_history.append(epoch_loss)
        print()

    time_elapsed = time.time() - since
    print('Training and Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'loss': val_loss_history}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, model_ver, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = getattr(models, f'{model_name}_{model_ver}')(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft

import timm
class HRnetModel(nn.Module):
    def __init__(self, num_classes=48, model_name=f'{model_name}_{model_ver}', pretrained=True):
        super(HRnetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x

# Initialize the model for this run
model_ft = HRnetModel()

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Print the model we just instantiated
print(model_ft)

# Data augmentation and normalization for training with Albumentations
A_transforms = {
    'train':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.HorizontalFlip(p=1),
            A.OneOf([
                     A.RandomRotate90(p=1),
                     A.VerticalFlip(p=1)
                     ], p=0.5),
            A.OneOf([A.MotionBlur(p=1),
                     A.GaussNoise(p=1)
                     ], p=0.5),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True,
                                            angle_in_degrees=True)),

    'val':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True,
                                            angle_in_degrees=True)),

    'test':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
}


class Dataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, data_dir, imgs, keypoints, phase, class_labels=None, data_transforms=None):
        self.data_dir = data_dir
        self.imgs = imgs
        self.keypoints = keypoints
        self.phase = phase
        self.class_labels = class_labels
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))
        keypoints = self.keypoints[idx]

        if self.data_transforms:
            augmented = self.data_transforms[self.phase](image=img, keypoints=keypoints, class_labels=self.class_labels)
            img = augmented['image']
            keypoints = augmented['keypoints']
        keypoints = np.array(keypoints).flatten()

        return img, keypoints

    def __len__(self):
        return len(self.imgs)

# Setup the loss fxn
criterion = nn.MSELoss()

since = time.time()
X_train, X_val, y_train, y_val = train_test_split(imgs, keypoints, test_size=1/num_splits, random_state=42)
train_data = Dataset(train_dir, X_train, y_train, data_transforms=A_transforms, class_labels=class_labels, phase='train')
val_data = Dataset(train_dir, X_val, y_val, data_transforms=A_transforms, class_labels=class_labels, phase='val')
train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}


# Observe that all parameters are being optimized
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_ft,
                                                  mode='min',
                                                  factor=0.5,
                                                  patience=2,)

# Train and evaluate
model_ft, hists = train_model(
    model_ft, dataloaders, criterion, optimizer_ft, lr_scheduler,
    num_epochs=num_epochs, earlystop=num_earlystop)
torch.save(model_ft.state_dict(), f'{prefix_dir}/hrnet_models/baseline_{model_name}{model_ver}.pt')
time_elapsed = time.time() - since
print('Elapsed time: {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))