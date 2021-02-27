import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import glob
import tensorflow as tf
from tensorflow.keras.models import load_model

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os

# 폴더 경로를 설정해줍니다.
os.chdir('./1. open')


save_model = load_model('./model/05-1061.1457.hdf5')

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
submission.to_csv('baseline_aug.csv', index=False)
print(submission.head())