import torch
import torch.nn as nn

from backbone import Backbone

class SSDHead(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(SSDHead, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.classification_layers = nn.ModuleList([
            nn.Conv2d(256, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, num_anchors * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, num_anchors * num_classes, kernel_size=3, padding=1)
        ])

        self.regression_layers = nn.ModuleList([
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, num_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, num_anchors * 4, kernel_size=3, padding=1)
        ])

    def forward(self, features):
        class_predictions = []
        box_regression = []

        for feature, cls_layer, reg_layer in zip(features, self.classification_layers, self.regression_layers):
            class_predictions.append(cls_layer(feature).permute(0, 2, 3, 1).contiguous())
            box_regression.append(reg_layer(feature).permute(0, 2, 3, 1).contiguous())

        class_predictions = torch.cat([pred.view(pred.size(0), -1, self.num_classes) for pred in class_predictions], dim=1)
        box_regression = torch.cat([reg.view(reg.size(0), -1, 4) for reg in box_regression], dim=1)

        return class_predictions, box_regression

class SSDModel(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super(SSDModel, self).__init__()

        self.backbone = Backbone()
        self.head = SSDHead(num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        class_predictions, box_regression = self.head(features)
        return class_predictions, box_regression
