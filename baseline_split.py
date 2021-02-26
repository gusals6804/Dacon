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

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import keras.backend as k

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

os.chdir('./1. open')

all_train = pd.read_csv('train_df.csv')

all_train.head(2)

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

print(train.shape, valid.shape)


def trainGenerator():
    # 원본 이미지 resize
    for i in range(len(train)):
        img = Image.open(train['path'][i])  # path(경로)를 통해 이미지 읽기
        img = img.resize([480, 270])  # 이미지 resize
        img = np.array(img)  # 넘파이 배열로 변환
        img = img / 255  # 이미지 rescaling
        target = train.iloc[:, 1:49].iloc[i, :]  # keypoint 뽑아주기
        target = target / 4  # image size를 1920x1080 -> 480x270으로 바꿔줬으므로 keypoint도 변경

        yield (img, target)


def validGenerator():
    # 원본 이미지 resize
    for i in range(len(valid)):
        img = Image.open(valid['path'][i])  # path(경로)를 통해 이미지 읽기
        img = img.resize([480, 270])  # 이미지 resize
        img = np.array(img)  # 넘파이 배열로 변환
        img = img / 255  # 이미지 rescaling
        target = valid.iloc[:, 1:49].iloc[i, :]  # keypoint 뽑아주기
        target = target / 4  # image size를 1920x1080 -> 480x270으로 바꿔줬으므로 keypoint도 변경

        yield (img, target)

batch_size = 32

train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([270,480,3]),tf.TensorShape([48])))
train_dataset = train_dataset.batch(batch_size).repeat().prefetch(1)
valid_dataset = tf.data.Dataset.from_generator(validGenerator, (tf.float32, tf.float32), (tf.TensorShape([270,480,3]),tf.TensorShape([48])))
valid_dataset = valid_dataset.batch(batch_size).repeat().prefetch(1)

#간단한 CNN 모델을 적용합니다.

model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(270,480,3)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(48))
model.summary()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

checkpointer = ModelCheckpoint(filepath = './model/{epoch:02d}-{val_mean_absolute_error:.4f}.hdf5', monitor='val_mean_absolute_error',
                               verbose=1, save_best_only=True, mode='min')
early_stopping_callback = EarlyStopping(monitor='val_mean_absolute_error', patience=5)


# model.fit(train_dataset, epochs=20, steps_per_epoch=len(train)//batch_size-1,
#            validation_data=valid_dataset, validation_steps=len(valid)//batch_size-1, callbacks=[early_stopping_callback,checkpointer])


save_model = load_model('./model/07-21.6712.hdf5')

test_paths = glob.glob('./test_imgs/*.jpg')
test_paths.sort()
X_test=[]

for test_path in tqdm(test_paths):
    img = tf.io.read_file(test_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [480, 270])
    img = img / 255
    X_test.append(img)

X_test=tf.stack(X_test, axis=0)
X_test.shape

# pred = save_model.predict(X_test, steps=1)
pred = []
i, chunksize = 0, 100
for idx in range(0, len(X_test), chunksize):
    pred += list(save_model.predict(X_test[idx:(i+1)*chunksize],
                verbose=1, batch_size=1))
    i += 1
pred = np.array(pred)

submission = pd.read_csv('./sample_submission.csv')
submission.iloc[:,1:] = pred * 4     # image size를 1920x1080 -> 480x270으로 바꿔서 예측했으므로 * 4
# submission

submission.to_csv('baseline.csv', index=False)


#
# # 예측 결과 시각화
# n = random.randint(0, 1600)
# predicted_keypoint = submission.iloc[n,1:49]
# predicted_keypoint = np.array(predicted_keypoint)
# img = Image.open(test_paths[n])
# plt.imshow(img)
# plt.scatter(predicted_keypoint[0::2], predicted_keypoint[1::2], marker='x')