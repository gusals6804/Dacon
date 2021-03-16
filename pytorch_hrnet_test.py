import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
import matplotlib.pyplot as plt
import imgaug
import glob
from typing import Tuple, List, Sequence, Callable, Dict

import albumentations
from albumentations.pytorch import ToTensorV2, ToTensor
from albumentations_experimental import HorizontalFlipSymmetricKeypoints

from torchvision import transforms
import timm

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# # 불러와서 예측하기
class HRnetModel(nn.Module):
    def __init__(self, num_classes=48, model_name='hrnet_w18', pretrained=True):
        super(HRnetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


PATH = './hrnet_models/1_hrnet_17.63908740874553.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HRnetModel().to(device)

model.load_state_dict(torch.load(PATH))


class TestDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, data_dir, imgs, data_transforms=None):
        self.data_dir = data_dir
        self.imgs = imgs
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.data_transforms:
            augmented = self.data_transforms(image=img)
            img = augmented['image']

        return filename, img

    def __len__(self):
        return len(self.imgs)


transforms_test = albumentations.Compose([
    albumentations.Crop(426, 56, 1450, 1080),
    albumentations.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
])


all_predictions = []
files = []



PATH_TEST_DATASET='./1. open/test_imgs/'
test_imgs = os.listdir(PATH_TEST_DATASET)
# Dataset 정의
test_dataset = TestDataset(PATH_TEST_DATASET, test_imgs, data_transforms=transforms_test)

# DataLoader 정의
test_data_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
)


with torch.no_grad():
    for filenames, inputs in tqdm(test_data_loader):
        predictions = list(model(inputs.to(device)).cpu().numpy())
        files.extend(filenames)
        for prediction in predictions:
            all_predictions.append(prediction)


SUB_DF = './1. open/sample_submission.csv'
submit = pd.read_csv(SUB_DF)
submit.head(2)

all_predictions = np.array(all_predictions)
for i in range(all_predictions.shape[0]):
    all_predictions[i, [2*j for j in range(24)]] *= (1920/512)
    all_predictions[i, [2*j + 1 for j in range(24)]] *= (1080/512)

submit = pd.DataFrame(columns=submit.columns)
submit['image'] = files
submit.iloc[:, 1:] = all_predictions
print(submit.head())

submit.to_csv('./submit_hr_0316.csv', index=False)

plt.figure(figsize=(40, 20))
count = 1
test_paths = glob.glob(os.path.join(PATH_TEST_DATASET, '*.jpg'))

for i in np.random.randint(0, len(test_paths), 5):

    plt.subplot(5, 1, count)

    img_sample_path = test_paths[i]
    print(img_sample_path.split('\\'))
    img_name = img_sample_path.split('\\')[1]
    print(img_name)
    img = Image.open(img_sample_path)
    img_np = np.array(img)
    key = submit[submit['image'] == img_name].iloc[0, 1:49]
    print(key)

    keypoint = submit.iloc[:, 1:49]  # 위치키포인트 하나씩 확인

    for j in range(0, len(keypoint.columns), 2):
        plt.plot(key[j], key[j + 1], 'rx')
        plt.show(img_np)

    count += 1