import os
from typing import Tuple, List, Sequence, Callable

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

import os
from typing import Tuple, List, Sequence, Callable, Dict

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import albumentations as A
from albumentations.pytorch import ToTensorV2

df = pd.read_csv('./1. open/train_df.csv')
df.head()
keypoint_names = df.columns[1:].tolist()

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (9, 18),
    (10, 19), (5, 7), (7, 9), (11, 13), (13, 15), (12, 14),
    (14, 16), (15, 22), (16, 23), (20, 21), (5, 6), (5, 11),
    (6, 12), (11, 12), (17, 20), (20, 21),
]

def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    edges: List[Tuple[int, int]] = None,
    keypoint_names: Dict[int, str] = None,
    boxes: bool = True,
    dpi: int = 200
) -> None:
    """
    Args:
        image (ndarray): [H, W, C]
        keypoints (ndarray): [N, 3]
        edges (List(Tuple(int, int))):
    """
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(24)}

    if boxes:
        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(
            image,
            tuple(keypoint),
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image,
                tuple(keypoints[edge[0]]),
                tuple(keypoints[edge[1]]),
                colors.get(edge[0]), 3, lineType=cv2.LINE_AA)

    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image)
    ax.axis('off')
    plt.show()
    fig.savefig('example.png')


def get_model() -> nn.Module:
    backbone = resnet_fpn_backbone('resnet152', pretrained=True)
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
            img = img/255.0

        return filename, img

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)
model.load_state_dict(torch.load('./1. open/rcnn_model/test-epoch-3-10.938045573339384.pt'))
model.eval()

all_predictions = []
files = []

device='cuda'

PATH_TEST_DATASET='./1. open/test_imgs/'
test_imgs = os.listdir(PATH_TEST_DATASET)

transforms_test = A.Compose([
    ToTensorV2()
])

# Dataset 정의
test_dataset = TestDataset(PATH_TEST_DATASET, test_imgs, data_transforms=transforms_test)
# DataLoader 정의
test_data_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)



count = 0
with torch.no_grad():
    for filenames, inputs in tqdm(test_data_loader):

        images = list(image.float().to(device) for image in inputs)
        predictions = model(images)
        files.extend(filenames)
        for prediction in predictions:
            try:
                predict = prediction['keypoints'][0][:, :2].cpu().numpy()
                predict = predict.reshape(-1)
                print(predict.shape)
                all_predictions.append(predict)
                print(np.array(all_predictions).shape)
            except:
                all_predictions.append(all_predictions[-1])
                count += 1
                pass

print(count)
all_predictions = np.array(all_predictions)
all_predictions = all_predictions.reshape(1600, 48)
print(all_predictions.shape)

SUB_DF = './1. open/sample_submission.csv'
submit = pd.read_csv(SUB_DF)
submit.head(2)

submit = pd.DataFrame(columns=submit.columns)
submit['image'] = files
submit.iloc[:, 1:] = all_predictions


print(submit.head())

submit.to_csv('./submit_rcnn.csv', index=False)


image = cv2.imread('./1. open/test_imgs/753-3-5-38-Z94_C-0000007.jpg', cv2.COLOR_BGR2RGB)
print(image)
image = image / 255.0
image = image.transpose(2, 0, 1)
image = [torch.as_tensor(image, dtype=torch.float32)]

model = get_model()
model.load_state_dict(torch.load('./1. open/rcnn_model/test-epoch-9-3.565001994968334.pt'))
model.eval()
preds = model(image)
keypoints = preds[0]['keypoints'].detach().numpy().copy()[0]
image = cv2.imread('./1. open//test_imgs/753-3-5-38-Z94_C-0000007.jpg', cv2.COLOR_BGR2RGB)
keypoints = keypoints[:, :2]

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (9, 18),
    (10, 19), (5, 7), (7, 9), (11, 13), (13, 15), (12, 14),
    (14, 16), (15, 22), (16, 23), (20, 21), (5, 6), (5, 11),
    (6, 12), (11, 12), (17, 20), (20, 21),
]

draw_keypoints(image, keypoints, edges, boxes=False)
