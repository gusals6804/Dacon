#!/usr/bin/env python
# coding: utf-8

# In[49]:


import os
from typing import Tuple, List, Sequence, Callable, Dict
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import random
import imgaug
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection.rpn import AnchorGenerator
from albumentations_experimental import HorizontalFlipSymmetricKeypoints
import timm

PATH_TRAIN_CSV = './1. open/train_df.csv'

train_df = pd.read_csv(PATH_TRAIN_CSV)
print(train_df.head(5))

error_list = [317, 869, 873, 877, 911, 1559, 1560, 1562, 1566, 1575, 1577, 1578, 1582, 1606, 1607, 1622,
              1623, 1624, 1625, 1629, 3968, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124,
              4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139,
              4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154,
              4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169,
              4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185,
              4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194]

# train, valid folder 속 모든 이미지 파일 read & sort
train_paths = glob.glob('./1. open/train_imgs/*.jpg')
train_paths.sort()
new_train_paths = []

add = 0
for i in range(len(train_paths)):
    print(train_paths[i])
    for j in error_list:
        if train_paths[i].split('\\')[-1] != train_df['image'][j]:
            add = 0
        elif train_paths[i].split('\\')[-1] == train_df['image'][j]:
            add = 1
            break
    if add == 0:
        new_train_paths.append(train_paths[i])
print(len(new_train_paths))

train_df.drop(error_list, inplace=True)
print(len(train_df))

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


class KeypointDataset(Dataset):
    def __init__(
            self,
            data_imgs_dir,
            keypoints,
            transforms: Sequence[Callable] = None
    ) -> None:
        self.data_imgs_dir = data_imgs_dir
        self.keypoints = keypoints
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data_imgs_dir)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        labels = np.array([1])
        keypoints = self.keypoints[index]

        x1, y1 = min(keypoints[:, 0]) - 5, min(keypoints[:, 1]) - 5
        x2, y2 = max(keypoints[:, 0]) + 5, max(keypoints[:, 1]) + 5
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = cv2.imread(self.data_imgs_dir.iloc[index].values[0], cv2.COLOR_BGR2RGB)
        # print(self.data_imgs_dir.iloc[index].values[0])
        # print(keypoints)

        targets = {
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'image_id': index,
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
            'keypoints': torch.as_tensor(
                np.concatenate([targets['keypoints'], np.ones((24, 1))], axis=1)[np.newaxis], dtype=torch.float32,
            ),
            'image_id': torch.as_tensor(targets['image_id'], dtype=torch.int64),
        }

        return image, targets


train_transforms = A.Compose([
    #A.Crop(426, 56, 1450, 1080),
    A.Resize(256, 256, always_apply=True),
    # A.OneOf([
    #     HorizontalFlipSymmetricKeypoints(symmetric_keypoints={
    #         'left': 'right'
    #     }, p=1),
    #     A.Rotate(limit=180, p=1),
    # ], p=1),
    # HorizontalFlipSymmetricKeypoints(symmetric_keypoints={
    #     'left': 'right'
    # }, p=1),
    # A.Rotate(limit=180, p=1),
    # A.OneOf([
    #     A.RandomRotate90(p=1),
    #     A.VerticalFlip(p=1)], p=0.5),
    # A.OneOf([
    #     A.MotionBlur(p=1),
    #     A.GaussNoise(p=1),
    # ], p=1),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    keypoint_params=A.KeypointParams(format='xy')
)

valid_transforms = A.Compose([
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    keypoint_params=A.KeypointParams(format='xy')
)


def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


PATH_TRAIN_CSV = './1. open/train_df.csv'

new_train_paths = pd.DataFrame(new_train_paths)
# trainset = KeypointDataset(new_train_paths, keypoints, train_transforms)
# train_loader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=collate_fn)


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


def get_model() -> nn.Module:

    backbone = resnet_fpn_backbone('resnet18', pretrained=True)
    #backbone = timm.create_model('hrnet_w48', pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    model = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=24
    )

    return model


# In[69]:
criterion = torch.nn.MSELoss()
PATH = './1. open/rcnn_model'


def train(device='cuda'):


    # cross validation을 적용하기 위해 KFold 생성
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    # dirty_mnist_answer에서 train_idx와 val_idx를 생성
    best_models = []  # 폴드별로 가장 validation acc가 높은 모델 저장
    for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(new_train_paths), 1):
        print(f'[fold: {fold_index}]')

        '''
        실제로 colab에서 학습할 때에는 시간 절약을 위해 fold별로 여러 session을 열어두고 병렬 작업을 수행했습니다.
        각각의 fold 작업에서 동일한 seed로 작업이 진행되기 때문에 이 코드에서 제출 결과를 재현하기 위해서는
        seed 초기화가 필요할 수 있습니다.
    
        seed_everything(config.seed)
        '''
        seed_everything(42)

        # cuda cache 초기화
        torch.cuda.empty_cache()

        # train fold, validation fold 분할
        train_paths = pd.DataFrame(new_train_paths)
        train_answer = train_paths.iloc[trn_idx[:2]]
        test_answer = train_paths.iloc[val_idx]

        train_keypoints = keypoints[trn_idx]
        test_keypoints = keypoints[val_idx]

        trainset = KeypointDataset(train_answer, train_keypoints, train_transforms)
        train_loader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        trainset = KeypointDataset(test_answer, test_keypoints, valid_transforms)
        valid_loader = DataLoader(trainset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        model = get_model()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

        # 학습률 스케쥴러를 만듭니다
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        num_epochs = 50
        valid_best_loss = 999999999
        #train_best_loss = 999999999
        early_stop_count = 0

        for epoch in range(num_epochs):

            model.train()
            train_running_loss = 0.0
            for i, (images, targets) in enumerate(train_loader):
                input_size = images[0].size(0)
                images = list(image.float().to(device) for image in images)
                # print(len(images))
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # print(len(targets))
                optimizer.zero_grad()
                # y_hat = model(images)
                # loss = criterion(y_hat, targets)
                # tensor를 gpu에 올리기

                losses = model(images, targets)

                loss = sum(loss for loss in losses.values())
                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print(f'| epoch: {epoch} | loss: {loss.item():.4f}', end=' | ')
                    for k, v in losses.items():
                        print(f'{k[5:]}: {v.item():.4f}', end=' | ')
                    print(i+1)
                    print()

                # statistics
                train_running_loss += loss.item() * input_size

            train_epoch_loss = train_running_loss / len(train_loader.dataset)

            print('Loss: {:.4f}'.format(train_epoch_loss))

            # 1개 epoch학습 후 Validation 점수 계산
            valid_running_loss = 0.0
            with tqdm(valid_loader,
                      total=valid_loader.__len__(),
                      unit="batch") as valid_bar:
                for i, (images, targets) in enumerate(valid_bar):
                    valid_bar.set_description(f"Valid Epoch {epoch}")
                    optimizer.zero_grad()
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # 모델의 dropoupt, batchnormalization를 eval모드로 설정
                    model.eval()

                    all_predictions = []
                    all_targets = []
                    # .forward()에서 중간 노드의 gradient를 계산
                    with torch.no_grad():
                        # validation loss만을 계산
                        outputs = model(images)

                        for prediction in outputs:
                            predict = prediction['keypoints'][0][:, :].cpu().numpy()
                            all_predictions.append(predict)

                        for prediction in targets:
                            predict = prediction['keypoints'][0][:, :].cpu().numpy()
                            all_targets.append(predict)

                        all_predictions = torch.as_tensor(all_predictions, dtype=torch.float32)
                        all_targets = torch.as_tensor(all_targets, dtype=torch.float32)
                        print(all_targets)
                        print(all_predictions)
                        print(np.array(targets).shape, np.array(outputs).shape)

                        #res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                        valid_loss = criterion(all_predictions, all_targets)

                    valid_bar.set_postfix(valid_loss=valid_loss.item())
                    # statistics
                valid_running_loss += valid_loss.item() * images.size(0)

            valid_epoch_loss = valid_running_loss / len(valid_loader.dataset)

            print('Loss: {:.4f}'.format(valid_epoch_loss))
            # Learning rate 조절
            lr_scheduler.step(valid_loss)

            if abs(valid_epoch_loss) < abs(valid_best_loss):
                valid_best_loss = valid_epoch_loss
                early_stop_count = 0
                torch.save(model.state_dict(), os.path.join(PATH, 'resnet101-epoch-{}-{}.pt'.format(epoch, train_epoch_loss)))
            else:
                early_stop_count += 1
                if early_stop_count > 5:
                    print(f'early stopped at epoch: {epoch}')
                    break


train()

# In[ ]:
