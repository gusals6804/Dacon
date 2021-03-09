import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
import os
import cv2
import random
import shutil
import glob
from math import cos, sin, pi

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from hrnet_keras import SE_HRNet

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 폴더 경로를 설정해줍니다.
os.chdir('./1. open')

# 해당 코드는 아래 Train, Valid Split 이후에 실행
train = pd.read_csv('./train.csv')
valid = pd.read_csv('./valid.csv')

train_paths = glob.glob('./train/*.jpg')
valid_paths = glob.glob('./valid/*.jpg')
test_paths = glob.glob('./test_imgs/*.jpg')
print(len(train_paths), len(valid_paths), len(test_paths))

train_paths.sort()
valid_paths.sort()
test_paths.sort()

train['path'] = train_paths
valid['path'] = valid_paths

pixel_shifts = [12, 24, 36, 48]
rotation_angles = [12, 24]
inc_brightness_ratio = 1.2
dec_brightness_ratio = 0.8
noise_ratio = 0.008

wid = 2048
height = 2048

# 좌우 반전
def left_right_flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=1)
    for idx, sample_keypoints in enumerate(keypoints):
        if idx % 2 == 0:
            flipped_keypoints.append(wid - sample_keypoints)
        else:
            flipped_keypoints.append(sample_keypoints)

    # left_right_keypoints_convert
    for i in range(8):
        flipped_keypoints[2 + (4 * i):4 + (4 * i)], flipped_keypoints[4 + (4 * i):6 + (4 * i)] = \
            flipped_keypoints[4+(4*i):6+(4*i)], flipped_keypoints[2+(4*i):4+(4*i)]
    flipped_keypoints[36:38], flipped_keypoints[38:40] = flipped_keypoints[38:40], flipped_keypoints[36:38]
    flipped_keypoints[44:46], flipped_keypoints[46:48] = flipped_keypoints[46:48], flipped_keypoints[44:46]

    return flipped_images, flipped_keypoints


# 수직/수평 동시 이동
# forloop에서 shift_x, shift_y 중 하나만 놓으면
# 수직 또는 수평 이동만 따로 시행 가능
def shift_images(images, keypoints):
    # tensor -> numpy
    shifted_images = []
    shifted_keypoints = []
    for shift in pixel_shifts:
        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
            # 이동할 matrix 생성
            M = np.float32([[1,0,shift_x],[0,1,shift_y]])
            shifted_keypoint = np.array([])
            shifted_x_list = np.array([])
            shifted_y_list = np.array([])
            # 이미지 이동
            shifted_image = cv2.warpAffine(images, M, (wid,height), flags=cv2.INTER_CUBIC)
            # 이동한만큼 keypoint 수정
            for idx, point in enumerate(keypoints):
                if idx%2 == 0:
                    shifted_keypoint = np.append(shifted_keypoint, point+shift_x)
                    shifted_x_list = np.append(shifted_x_list, point+shift_x)
                else:
                    shifted_keypoint =np.append(shifted_keypoint, point+shift_y)
                    shifted_y_list = np.append(shifted_y_list, point+shift_y)
            # 수정된 keypoint가 이미지 사이즈를 벗어나지 않으면 append
            if np.all(0.0<shifted_x_list) and np.all(shifted_x_list<wid) and \
                    np.all(0.0<shifted_y_list) and np.all(shifted_y_list<height):
                shifted_images.append(shifted_image.reshape(height,wid,3))
                shifted_keypoints.append(shifted_keypoint)

    return shifted_images, shifted_keypoints



# 이미지 회전
def rotate_augmentation(images, keypoints):
    # tensor -> numpy
    rotated_images = []
    rotated_keypoints = []

    for angle in rotation_angles:
        for angle in [angle, -angle]:
            # 회전할 matrix 생성
            M = cv2.getRotationMatrix2D((240, 135), angle, 1.0)
            # cv2_imshow로는 문제없지만 추후 plt.imshow로 사진을 확인할 경우 black screen 생성...
            # 혹시 몰라 matrix를 ndarray로 변환
            M = np.array(M, dtype=np.float32)
            angle_rad = -angle * pi / 180
            rotated_image = cv2.warpAffine(images, M, (wid, height))
            rotated_images.append(rotated_image)

            # keypoint를 copy하여 forloop상에서 값이 계속 없데이트 되는 것을 회피
            rotated_keypoint = keypoints.copy()
            rotated_keypoint[0::2] = rotated_keypoint[0::2] - 240
            rotated_keypoint[1::2] = rotated_keypoint[1::2] - 135

            for idx in range(0, len(rotated_keypoint), 2):
                rotated_keypoint[idx] = rotated_keypoint[idx] * cos(angle_rad) - rotated_keypoint[idx + 1] * sin(
                    angle_rad)
                rotated_keypoint[idx + 1] = rotated_keypoint[idx] * sin(angle_rad) + rotated_keypoint[idx + 1] * cos(
                    angle_rad)

            rotated_keypoint[0::2] = rotated_keypoint[0::2] + 240
            rotated_keypoint[1::2] = rotated_keypoint[1::2] + 135
            rotated_keypoints.append(rotated_keypoint)

    return rotated_images, rotated_keypoints

# 이미지 해상도 조절
def alter_brightness(images):
    altered_brightness_images = []
    inc_brightness_images = np.clip(images*inc_brightness_ratio, 0.0, 1.0)
    dec_brightness_images = np.clip(images*dec_brightness_ratio, 0.0, 1.0)
    altered_brightness_images.append(inc_brightness_images)
    altered_brightness_images.append(dec_brightness_images)
    return altered_brightness_images

# Random 노이즈 추가
def add_noise(images):
    images = np.array(images)
    noise = noise_ratio * np.random.randn(wid, height, 3)
    noise = noise.astype(np.float32)
    # 생성한 noise를 원본에 add
    noisy_image = cv2.add(images, noise)
    return noisy_image


def trainGenerator():
    # 원본 이미지 resize
    for i in range(len(train)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255  # 이미지 rescaling
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)

        yield (img, target)

    # # horizontal flip
    for i in range(len(train)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)

        yield (img, target)
    #
    # Horizontal & Vertical shift
    for i in range(len(train)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)
        img_list, target_list = shift_images(img, target)
        for shifted_img, shifted_target in zip(img_list, target_list):
            yield (shifted_img, shifted_target)
    #
    # Rotation
    for i in range(len(train)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)
        img_list, target_list = rotate_augmentation(img, target)
        for rotated_img, rotated_target in zip(img_list, target_list):
            yield (rotated_img, rotated_target)
    #
    # Alter_Brightness
    for i in range(len(train)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)
        img_list = alter_brightness(img)
        for altered_brightness_images in img_list:
            yield (altered_brightness_images, target)


    #Adding_Noise
    for i in range(len(train)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)
        noisy_img = add_noise(img)

        yield (noisy_img, target)

def validGenerator():
    # 원본 이미지 resize
    for i in range(len(valid)):
        img = cv2.imread(train['path'][i], cv2.COLOR_BGR2RGB)  # path(경로)를 통해 이미지 읽
        img = cv2.resize(img, (wid, height))
        img = img / 255
        target = train.iloc[i, 1:49].values.reshape(-1, 2)
        target[:, 0] *= (wid / 1920)
        target[:, 1] *= (height / 1080)
        target = target.reshape(-1)

        yield (img, target)

batch_size = 16

train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([height,wid,3]),tf.TensorShape([48])))
train_dataset = train_dataset.batch(batch_size).repeat().prefetch(1)
valid_dataset = tf.data.Dataset.from_generator(validGenerator, (tf.float32, tf.float32), (tf.TensorShape([height,wid,3]),tf.TensorShape([48])))
valid_dataset = valid_dataset.batch(batch_size).repeat().prefetch(1)

# Callback 설정
earlystop = EarlyStopping(patience=7)
learning_rate_reduction=ReduceLROnPlateau(
                        monitor="val_loss",
                        patience=2,
                        factor=0.85,
                        min_lr=1e-7,
                        verbose=1)

model_check = ModelCheckpoint(  # 에포크마다 현재 가중치를 저장
        filepath='./model/{epoch:02d}-{val_loss:.4f}.hdf5',  # 모델 파일 경로
        monitor='val_loss',  # val_loss 가 좋아지지 않으면 모델 파일을 덮어쓰지 않음.
        save_best_only=True)

callbacks = [earlystop, model_check, learning_rate_reduction]

# Model Structure
se_HRNet = SE_HRNet(blocks=3, reduction_ratio=4, init_filters=24, training=True
                      ).build(input_shape=(wid, height, 3), num_output=48, repetitions=4)
# layer 동결 해제(일부 or 전체)

from tensorflow.keras import models
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.regularizers import l2

resnet152 = ResNet50(weights ='imagenet', include_top = False,
                       input_shape = (wid,height,3))

for layer in se_HRNet.layers:

    layer.trainable = True
    print(layer)

model = models.Sequential()
model.add(se_HRNet)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=48,
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1e-4),
                  activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])


history = model.fit(train_dataset,
                    epochs=1000,
                    steps_per_epoch=len(train)//batch_size-1,
                    validation_data=valid_dataset,
                    validation_steps=len(valid)//batch_size - 1,
                    callbacks=callbacks,
                    verbose=1)