import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        outputs.append(x.clone())
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        outputs.append(x.clone())
        x = self.maxpool5(self.relu5(self.bn5(self.conv5(x))))
        outputs.append(x.clone())
        return outputs
