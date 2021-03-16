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

import albumentations
from albumentations.pytorch import ToTensorV2, ToTensor
from albumentations_experimental import HorizontalFlipSymmetricKeypoints

from torchvision import transforms
import timm

model_names = timm.list_models(pretrained=True)
print(model_names)
# random seed
seed = 301

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 3
NUM_EPOCHS = 30

train_batch_size = 3
valid_batch_size = 3
test_batch_size = 16

device = "cuda"

# Number of classes in the dataset
num_classes = 48

# Learning rate for optimizer
lr = 1e-3

# fold
folds = 5

PATH_TRAIN_DATASET = './1. open/train_imgs/'
PATH_TEST_DATASET = './1. open/test_imgs/'
PATH_TRAIN_CSV = './1. open/train_df.csv'

train_df = pd.read_csv(PATH_TRAIN_CSV)
print(train_df.head())

# device 설정
device = torch.device(device if torch.cuda.is_available() else "cpu")


# seed 설정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    imgaug.random.seed(seed)


seed_everything(seed)

KEYPOINT_COLOR = (0, 255, 0)  # Green


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=3):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)


# train, valid folder 속 모든 이미지 파일 read & sort
train_paths = glob.glob('./1. open/train_imgs/*.jpg')
train_paths.sort()

imgs = train_df.iloc[:, 0].to_numpy()
keypoint = train_df.iloc[:, 1:]
columns = keypoint.columns.to_list()[::2]
class_labels = [label.replace('_x', '').replace('_y', '') for label in columns]
keypoints = []
for keypoint in keypoint.to_numpy():
    a_keypoints = []
    for i in range(0, keypoint.shape[0], 2):
        a_keypoints.append((float(keypoint[i]), float(keypoint[i + 1])))
    keypoints.append(a_keypoints)
keypoints = np.array(keypoints)
print(len(keypoints))


class MotionDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, data_imgs_dir, keypoints, class_labels=None, data_transforms=None):
        self.data_imgs_dir = data_imgs_dir
        self.keypoints = keypoints
        self.class_labels = class_labels
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        # Read an image with OpenCV
        img = cv2.imread(self.data_imgs_dir.iloc[idx].values[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        keypoints = self.keypoints[idx]

        if self.data_transforms:
            augmented = self.data_transforms(image=img, keypoints=keypoints, class_labels=self.class_labels)
            img = augmented['image']
            keypoints = augmented['keypoints']
            # keypoints = [(x / 512, y / 512) for x, y in keypoints]

        keypoints = np.array(keypoints).flatten()

        return img, keypoints

    def __len__(self):
        return len(self.data_imgs_dir)


transforms_train = albumentations.Compose([

    albumentations.Crop(426, 56, 1450, 1080),
    #albumentations.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
    albumentations.OneOf([
        HorizontalFlipSymmetricKeypoints(symmetric_keypoints={
            'left': 'right'
        }, p=1),
        albumentations.Rotate(limit=180, p=1),
    ], p=1),
    albumentations.OneOf([
        albumentations.RandomBrightness(p=1),
        albumentations.MotionBlur(blur_limit=[3, 20], p=1),
    ], p=0.5),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
], keypoint_params=albumentations.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))

transforms_valid = albumentations.Compose([

    albumentations.Crop(426, 56, 1450, 1080),
    #albumentations.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
    albumentations.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
], keypoint_params=albumentations.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=False))

path = './hrnetv2_w48 - abd2e6ab.pth'


class HRnetModel(nn.Module):
    def __init__(self, num_classes=48, model_name='hrnet_w64', pretrained=True):
        super(HRnetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.model(x)

        return x


# cross validation을 적용하기 위해 KFold 생성
from sklearn.model_selection import KFold

kfold = KFold(n_splits=folds, shuffle=True, random_state=0)

# dirty_mnist_answer에서 train_idx와 val_idx를 생성
best_models = []  # 폴드별로 가장 validation acc가 높은 모델 저장
for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(train_paths), 1):
    print(f'[fold: {fold_index}]')
    '''
    실제로 colab에서 학습할 때에는 시간 절약을 위해 fold별로 여러 session을 열어두고 병렬 작업을 수행했습니다.
    각각의 fold 작업에서 동일한 seed로 작업이 진행되기 때문에 이 코드에서 제출 결과를 재현하기 위해서는
    seed 초기화가 필요할 수 있습니다.

    seed_everything(config.seed)
    '''
    seed_everything(seed)
    # cuda cache 초기화
    torch.cuda.empty_cache()
    print(trn_idx)
    # train fold, validation fold 분할
    train_paths = pd.DataFrame(train_paths)
    train_answer = train_paths.iloc[trn_idx]
    print(train_answer)
    test_answer = train_paths.iloc[val_idx]

    train_keypoints = keypoints[trn_idx]
    print(len(train_keypoints))
    test_keypoints = keypoints[val_idx]

    # Dataset 정의
    train_dataset = MotionDataset(train_answer, train_keypoints, data_transforms=transforms_train,
                                  class_labels=class_labels)
    valid_dataset = MotionDataset(test_answer, test_keypoints, data_transforms=transforms_valid,
                                  class_labels=class_labels)

    # DataLoader 정의
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
    )

    # 모델 선언

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRnetModel().to(device)

    #print(summary(model, input_size=(1, 3, 512, 512), verbose=0))

    # 훈련 옵션 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
    criterion = torch.nn.MSELoss()

    # 훈련 시작
    valid_acc_max = 0
    valid_best_loss = 999999999
    early_stop_count = 0

    for epoch in range(NUM_EPOCHS):
        # 1개 epoch 훈련
        with tqdm(train_data_loader,  # train_data_loader를 iterative하게 반환
                  total=train_data_loader.__len__(),  # train_data_loader의 크기
                  unit="batch") as train_bar:  # 한번 반환하는 smaple의 단위는 "batch"
            train_running_loss = 0.0
            for i, (images, targets) in enumerate(train_bar):
                train_bar.set_description(f"Train Epoch {epoch}")
                # 갱신할 변수들에 대한 모든 변화도를 0으로 초기화
                # 참고)https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
                optimizer.zero_grad()
                # tensor를 gpu에 올리기
                images = images.to(device)
                targets = targets.to(device)

                # 모델의 dropoupt, batchnormalization를 train 모드로 설정
                model.train()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.set_grad_enabled(True):
                    # 모델 예측
                    output = model(images)
                    # loss 계산
                    loss = criterion(output.float(), targets.float())

                    # 중간 노드의 gradient로
                    # backpropagation을 적용하여
                    # gradient 계산
                    loss.backward()
                    # weight 갱신
                    optimizer.step()

                # 현재 progress bar에 현재 미니배치의 loss 결과 출력
                train_bar.set_postfix(train_loss=loss.item())

            # statistics
            train_running_loss += loss.item() * images.size(0)

        train_epoch_loss = train_running_loss / len(train_data_loader.dataset)

        print('Loss: {:.4f}'.format(train_epoch_loss))

        # 1개 epoch학습 후 Validation 점수 계산
        valid_running_loss = 0.0
        with tqdm(valid_data_loader,
                  total=valid_data_loader.__len__(),
                  unit="batch") as valid_bar:
            for i, (images, targets) in enumerate(valid_bar):
                valid_bar.set_description(f"Valid Epoch {epoch}")
                optimizer.zero_grad()
                images = images.to(device)
                targets = targets.to(device)

                # 모델의 dropoupt, batchnormalization를 eval모드로 설정
                model.eval()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.no_grad():
                    # validation loss만을 계산
                    probs = model(images)
                    valid_loss = criterion(probs.float(), targets.float())

                valid_bar.set_postfix(valid_loss=valid_loss.item())
                # statistics
            valid_running_loss += valid_loss.item() * images.size(0)

        valid_epoch_loss = valid_running_loss / len(valid_data_loader.dataset)

        print('Loss: {:.4f}'.format(valid_epoch_loss))
        # Learning rate 조절
        lr_scheduler.step()

        # 모델 저장
        if abs(valid_epoch_loss) < abs(valid_best_loss):
            valid_best_loss = valid_epoch_loss
            early_stop_count = 0
            best_model = model
            MODEL = 'hrnet'
            # 모델을 저장할 구글 드라이브 경로
            path = './hrnet_models/'
            torch.save(best_model.state_dict(), f'{path}{fold_index}_{MODEL}_{valid_epoch_loss}.pth')
        else:
            early_stop_count += 1
            if early_stop_count > 5:
                print(f'early stopped at epoch: {epoch}')
                break

    # 폴드별로 가장 좋은 모델 저장
    best_models.append(best_model)

#
