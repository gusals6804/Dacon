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
import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import imgaug
import matplotlib.pyplot as plt

input_w = 1333
input_h = 800

test_batch_size = 16
num_workers = 0

test_paths = glob.glob('./1. open/test_imgs/*.jpg')
test_paths.sort()

# Data augmentation and normalization for training with Albumentations
A_transforms = {

    'test':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
}




SAMPLE_SUBMISSION_FILE = './1. open/sample_submission.csv'
ADAM_MODEL_PATH = './1. open/adam_model/'


class TestDataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, data_img_dir, phase, data_transforms=None):
        self.data_img_dir = data_img_dir
        self.phase = phase
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        filename = self.data_img_dir[idx]
        # Read an image with OpenCV
        img = cv2.imread(self.data_img_dir[idx])

        if self.data_transforms:
            augmented = self.data_transforms[self.phase](image=img)
            img = augmented['image']

        sample = {'image': img}
        return img

    def __len__(self):
        return len(self.data_img_dir)


test_dataset = TestDataset(test_paths, phase='test', data_transforms=A_transforms)
#test Dataset 정의
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
test_data_loader = DataLoader(
    test_dataset,
    batch_size = test_batch_size,
    shuffle = False,
    num_workers = num_workers,
    drop_last = False
)



class AdamModel(nn.Module):
    def __init__(self):
        super(AdamModel, self).__init__()
        self.conv2d = nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.efn_b3 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1024)
        self.fc = nn.ModuleList([nn.Linear(1024, 1) for i in range(48)])

        nn.init.xavier_uniform_(self.conv2d.weight)
        for i in range(48):
            nn.init.xavier_uniform_(self.fc[i].weight)

    def forward(self, x):
        x = F.relu(self.conv2d(x))
        x = F.relu(self.efn_b3(x))
        xs = []
        for i in range(48):
            xs.append(self.fc[i](x))
        x = torch.cat(xs, dim=1)
        x = torch.sigmoid(x)
        return x

model = AdamModel()
model_files = os.listdir(ADAM_MODEL_PATH)
best_models = [torch.load(ADAM_MODEL_PATH + model_file) for model_file in model_files]

device = "cuda"
tta = True
tta_count = 32

predictions_list = []
# 배치 단위로 추론
prediction_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)

# 5개의 fold마다 가장 좋은 모델을 이용하여 예측
for model_index, model in enumerate(best_models):
    print(f'[model: {model_files[model_index]}]')
    if tta:
        count = tta_count
    else:
        count = 1
    for c in range(count):
        # 0으로 채워진 array 생성
        prediction_array = np.zeros([prediction_df.shape[0],
                                     prediction_df.shape[1] - 1])
        print(prediction_array.shape)
        with tqdm(test_data_loader,
                  total=test_data_loader.__len__(),
                  unit="batch") as test_bar:
            for idx, sample in enumerate(test_bar):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = sample['image']
                    images = images.to(device)
                    probs = model(images)
                    probs = probs.cpu().detach().numpy()

                    # idx = 1
                    # plt.imshow(images[idx][0])
                    # plt.title("sample input image")
                    # for j in range(0, len(probs[0].columns), 2):
                    #     plt.plot(probs[0][j], probs[0][j + 1], 'rx')
                    #
                    #
                    # plt.show()

                    # 예측 결과를
                    # prediction_array에 입력
                    batch_index = test_batch_size * idx
                    prediction_array[batch_index: batch_index + images.shape[0], :] \
                        = probs

        # 채널을 하나 추가하여 list에 append
        predictions_list.append(prediction_array[..., np.newaxis])

# axis = 2를 기준으로 평균
predictions_array = np.concatenate(predictions_list, axis = 2)
predictions_mean = predictions_array.mean(axis = 2)

# 평균 값이 0.5보다 클 경우 1 작으면 0
predictions_mean = (predictions_mean > 0.5) * 1
predictions_mean

SUBMISSION_FILE = './pytorch.csv'
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
sample_submission.iloc[:,1:] = predictions_mean
sample_submission.to_csv(SUBMISSION_FILE, index = False)
sample_submission