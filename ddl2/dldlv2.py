import torch
import PIL
import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from scipy.stats import norm
import torch.nn.functional as F
import tqdm


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.filenames = [os.path.join(path, name) for path, subdirs, files in os.walk(self.images_folder) for name in files]
        frames_list = [f'frame_00{i}0' for i in range(10)]
        self.filenames = [x for x in self.filenames for frame in frames_list if frame in x]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return filename, image


transform_size_normalized = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4816, 0.4199, 0.3884], [0.1327, 0.1273, 0.1356])
])

test_data = CustomDataset(r"C:\Users\Lenovo\PycharmProjects\avantgarde\data\facebook_frames\\", transform_size_normalized)
test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=False)


def Conv(in_channels, out_channels, kerner_size=3, stride=1, padding=1):
    out_channels = int(out_channels)
    in_channels = int(in_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kerner_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


class DLDLv2(nn.Module):
    def __init__(self, max_age=101, c=0.5):
        super(DLDLv2, self).__init__()
        self.conv1 = Conv(3, 64 * c)
        self.conv2 = Conv(64 * c, 64 * c)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(64 * c, 128 * c)
        self.conv4 = Conv(128 * c, 128 * c)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(128 * c, 256 * c)
        self.conv6 = Conv(256 * c, 256 * c)
        self.conv7 = Conv(256 * c, 256 * c)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = Conv(256 * c, 512 * c)
        self.conv9 = Conv(512 * c, 512 * c)
        self.conv10 = Conv(512 * c, 512 * c)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv11 = Conv(512 * c, 512 * c)
        self.conv12 = Conv(512 * c, 512 * c)
        self.conv13 = Conv(512 * c, 512 * c)

        self.HP = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(int(512 * c), max_age),
            nn.Sigmoid()
        )

        self.ages = torch.tensor(list(range(max_age))).t().float()
        self.device = "cpu"

        self.transform_img = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0.4816, 0.4199, 0.3884], [0.1327, 0.1273, 0.1356])
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.HP(x)
        x = x.view((x.size(0), -1))  # flatten layer x.size(0) is batchsize
        x = self.fc1(x.view((x.size(0), -1)))
        x = F.normalize(x, p=1, dim=1)
        return x

    def to(self, device):
        module = super(DLDLv2, self).to(device)
        module.ages = self.ages.to(device)
        self.device = device
        return module

    # predict age of a batch
    def predict_age(self, x):
        x = x.to(device)
        with torch.no_grad():
            outputs = self.forward(x)

        return torch.matmul(outputs, self.ages)

    def predict_age_logits(self, x):
        x = x.to(device)
        with torch.no_grad():
            outputs = self.forward(x)

        return torch.matmul(outputs, self.ages), outputs

    def predict_age_and_transform(self, x):
        x = self.transform_img(x)
        x = x.to(device)

        with torch.no_grad():
            outputs = self.forward(x)

        return torch.matmul(outputs, self.ages), outputs

model = DLDLv2(101, 0.5)
model.load_state_dict(torch.load("ThinAgeNet-ChaLearn15.pt"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.to("cuda")


results = []
model.eval()
print(len(test_dataloader))


for filename, image in tqdm.tqdm(test_dataloader):
    try:
        image = image.to(device)
        age = model.predict_age(image)
        results.append((filename[0], age[0].item()))
    except:
        continue
df = pd.DataFrame(results)
df.to_csv('dldlv2_frames_results.csv')