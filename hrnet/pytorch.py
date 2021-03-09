#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[1]:


import numpy as np
import pandas as pd
import os
import time
import copy
import random
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from hrnet import HRNet

from sklearn.model_selection import train_test_split

# For image-keypoints data augmentation

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import imgaug

# In[105]:


train_df_path = './1. open/train_df.csv'

seed = 42
device = "cuda"    

# Number of classes in the dataset
num_classes = 48

# Learning rate for optimizer
lr = 1e-3

# Number of epochs and earlystop to train for
num_epochs = 50
train_batch_size = 2
valid_batch_size = 2
test_batch_size = 16
num_workers = 0

ADAM_MODEL_PATH = './1. open/adam_model/'
#SAM_MODEL_PATH = './models/sam_model/'
ADAM_MODEL_PREFIX = 'eff_b3_adam_ind'
#SAM_MODEL_PREFIX = 'eff_b3_sam_ind'

tta = True
tta_count = 32

folds = 5

# Iput size for resize imgae
input_w = 512
input_h = 512


# In[106]:


train_df = pd.read_csv(train_df_path)
train_df.head()


# In[107]:


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


# In[108]:


KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=3):
    image = image.copy()

    for (x, y) in keypoints:
        
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)


# In[109]:


# train, valid folder 속 모든 이미지 파일 read & sort
train_paths = glob.glob('./1. open/train_imgs/*.jpg')
train_paths.sort()


# In[110]:


imgs = train_df.iloc[:, 0].to_numpy()
keypoint = train_df.iloc[:, 1:] 
columns = keypoint.columns.to_list()[::2]
class_labels = [label.replace('_x', '').replace('_y', '') for label in columns]
keypoints = []
for keypoint in keypoint.to_numpy():
    a_keypoints = []
    for i in range(0,keypoint.shape[0], 2):
        a_keypoints.append((float(keypoint[i]), float(keypoint[i+1])))
    keypoints.append(a_keypoints)
keypoints = np.array(keypoints)
print(len(keypoints))


# In[111]:


# for i in np.random.randint(0,len(train_paths),5):
#     image = cv2.imread(train_paths[i])
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     sample_keypoint = keypoints[i]
#     vis_keypoints(image, sample_keypoint)
    
#     x_min, y_min = int(min(sample_keypoint[:, 0])/2)-10, int(min(sample_keypoint[:, 1])/2)-10
#     x_max, y_max = int(max(sample_keypoint[:, 0])/2)+10, int(max(sample_keypoint[:, 1])/2)+10
#     transform = A.Compose(
#         [A.Resize(input_h, input_w, always_apply=True),
#          A.CenterCrop(height=420, width=420, p=1),
#         ], 
#         keypoint_params=A.KeypointParams(format='xy')
#     )
#     transformed = transform(image=image, keypoints=sample_keypoint)
#     vis_keypoints(transformed['image'], transformed['keypoints'])


# In[112]:


# unmpy를 tensor로 변환하는 ToTensor 정의
class ToTensor(object):
    """numpy array를 tensor(torch)로 변환합니다."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        print(image.shape)
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}
# to_tensor 선언
to_tensor = T.Compose([
                        ToTensorV2()
                    ])


# In[113]:


A_transforms  = {
            'train':
                A.Compose([
                    A.Resize(input_h, input_w, always_apply=True),
                    A.OneOf([A.HorizontalFlip(p=1),
                             A.RandomRotate90(p=1),
                             A.VerticalFlip(p=1)], p=0.5),
                    A.OneOf([A.MotionBlur(p=1),
                             A.GaussNoise(p=1)], p=0.5),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ToTensorV2()
                    ],  
                    keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'])
                    ),
                
            'val':
                A.Compose([
                    A.Resize(input_h, input_w, always_apply=True),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
                
            }


# In[114]:


class Dataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_img_dir, keypoints, phase, class_labels=None,  data_transforms=None, transforms=to_tensor):
        self.data_img_dir = data_img_dir
        self.keypoints = keypoints
        self.class_labels = class_labels
        self.phase = phase
        self.transforms = transforms# Transform
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        # Read an image with OpenCV
        img = cv2.imread(self.data_img_dir.iloc[idx].values[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        keypoints = self.keypoints[idx]


        #x_min, y_min = int(min(keypoints[:, 0])/2)-10, int(min(keypoints[:, 1])/2)-10
        #x_max, y_max = int(max(keypoints[:, 0])/2)+10, int(max(keypoints[:, 1])/2)+10
        
        #image = cv2.resize(image, dsize=(config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        if self.data_transforms:
    
            augmented = self. data_transforms[self.phase](image=img, keypoints=keypoints, class_labels=self.class_labels)
            img = augmented['image']
            keypoints = augmented['keypoints']
        
        #print(img.shape)
        #img = (img/255).astype('float')[..., np.newaxis]

        
        keypoints = np.array(keypoints).flatten()
        #keypoints = keypoints.reshape(-1, 2)
        sample = {'image': img, 'label': keypoints}
        print(img.shape, keypoints.shape)

        # transform 적용
#         # numpy to tensor
#         if self.transforms:
#             sample = self.transforms(sample)
#         # sample 반환
        return sample
    
    def __len__(self):
        return len(self.data_img_dir)


# In[115]:
model_c=48
model_nof_joints=24
model_bn_momentum=0.1


# class AdamModel(nn.Module):
#     def __init__(self):
#         super(AdamModel, self).__init__()
#         self.conv2d = nn.Conv2d(3, 3, 3, stride=1, padding=1)
#         self.efn_b3 = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1024)
#         self.fc = nn.ModuleList([nn.Linear(1024, 1) for i in range(48)])
#
#         nn.init.xavier_uniform_(self.conv2d.weight)
#         for i in range(48):
#             nn.init.xavier_uniform_(self.fc[i].weight)
#
#     def forward(self, x):
#         x = F.relu(self.conv2d(x))
#         x = F.relu(self.efn_b3(x))
#         xs = []
#         for i in range(48):
#             xs.append(self.fc[i](x))
#         x = torch.cat(xs, dim=1)
#         x = torch.sigmoid(x)
#         return x
#
#
# # In[116]:
#
#
# from typing import Tuple, List, Sequence, Callable
# def collate_fn(batch: torch.Tensor):
#     return zip(*batch)


# In[117]:

def flip_tensor(tensor, dim=0):
    """
    flip the tensor on the dimension dim
    """
    inv_idx = torch.arange(tensor.shape[dim] - 1, -1, -1).to(tensor.device)
    return tensor.index_select(dim, inv_idx)

def flip_back(output_flipped):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'

    output_flipped = flip_tensor(output_flipped, dim=-1)

    # for pair in matched_parts:
    #     tmp = output_flipped[:, pair[0]].clone()
    #     output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
    #     output_flipped[:, pair[1]] = tmp

    return output_flipped

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        """
        MSE loss between output and GT body joints
        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
                This has not the same meaning of joint_weights in the COCO dataset.
        """
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        print(output.shape)
        print(heatmaps_pred )
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        print(heatmaps_gt)
        print(target.reshape)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                if target_weight is None:
                    raise NameError
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

# cross validation을 적용하기 위해 KFold 생성
from sklearn.model_selection import KFold
kfold = KFold(n_splits=folds, shuffle=True, random_state=0)

# dirty_mnist_answer에서 train_idx와 val_idx를 생성
best_models = [] # 폴드별로 가장 validation acc가 높은 모델 저장
for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(train_paths),1):
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
    #train fold, validation fold 분할
    train_paths = pd.DataFrame(train_paths)
    train_answer = train_paths.iloc[trn_idx]
    test_answer  = train_paths.iloc[val_idx]
    
    train_keypoints = keypoints[trn_idx]
    print(len(train_keypoints))
    test_keypoints = keypoints[val_idx]

    #Dataset 정의
    train_dataset = Dataset(train_answer, train_keypoints, data_transforms=A_transforms, class_labels=class_labels,  phase='train')
    valid_dataset = Dataset(test_answer, test_keypoints, data_transforms=A_transforms, class_labels=class_labels, phase='val')

    #DataLoader 정의
    train_data_loader = DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size = valid_batch_size,
        shuffle = False,
        num_workers = num_workers,
    )

    # 모델 선언
    model = HRNet(c=model_c, nof_joints=model_nof_joints,
                  bn_momentum=model_bn_momentum).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)# gpu에 모델 할당

    # 훈련 옵션 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
    criterion = torch.nn.BCELoss()

    # 훈련 시작
    valid_acc_max = 0
    valid_best_loss = 999999999
    early_stop_count = 0
    for epoch in range(num_epochs):
        # 1개 epoch 훈련
        with tqdm(train_data_loader,  # train_data_loader를 iterative하게 반환
                  total=train_data_loader.__len__(),  # train_data_loader의 크기
                  unit="batch") as train_bar:  # 한번 반환하는 smaple의 단위는 "batch"
            train_running_loss = 0.0
            for sample in train_bar:
                train_bar.set_description(f"Train Epoch {epoch}")
                # 갱신할 변수들에 대한 모든 변화도를 0으로 초기화
                # 참고)https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
                optimizer.zero_grad()
                images, labels = sample['image'], sample['label']
                print(labels.shape)
                # tensor를 gpu에 올리기
                images = images.to(device)
                labels = labels.to(device)

                # 모델의 dropoupt, batchnormalization를 train 모드로 설정
                model.train()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.set_grad_enabled(True):
                    # 모델 예측
                    output = model(images)
                    print(output.shape)
                    # image_flipped = flip_tensor(images, dim=-1)
                    # print(image_flipped.shape)
                    #
                    # output_flipped = model(image_flipped)
                    #
                    # output_flipped = flip_back(output_flipped)
                    #
                    # output = (output + output_flipped) * 0.5

                    print(output.shape, labels.shape)
                    # loss 계산
                    loss_fn = JointsMSELoss().to(device)
                    loss = loss_fn(output, labels)
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
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch {epoch}")
                optimizer.zero_grad()
                images, labels = sample['image'], sample['label']
                images = images.to(device)
                labels = labels.to(device)

                # 모델의 dropoupt, batchnormalization를 eval모드로 설정
                model.eval()
                # .forward()에서 중간 노드의 gradient를 계산
                with torch.no_grad():
                    # validation loss만을 계산
                    probs = model(images)
                    valid_loss = criterion(probs.float(), labels.float())

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
            MODEL = ADAM_MODEL_PREFIX
            # 모델을 저장할 구글 드라이브 경로
            path = ADAM_MODEL_PATH
            torch.save(best_model, f'{path}{fold_index}_{MODEL}_{valid_epoch_loss}.pth')
        else:
            early_stop_count += 1
            if early_stop_count > 5:
                print(f'early stopped at epoch: {epoch}')
                break

    # 폴드별로 가장 좋은 모델 저장
    best_models.append(best_model)


# In[ ]:





# In[ ]:





# In[ ]:




