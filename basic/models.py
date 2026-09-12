#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.


import torch
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d
from torch.nn import InstanceNorm2d
from torch.nn import Dropout, Dropout2d
from torch.nn import ReLU
from torch.nn.functional import pad
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random


import torchvision.models as models
import torch.nn as nn

class ResNet18CTC(nn.Module):
    def __init__(self, params):
        super().__init__()

        out_channels = params["hidden_size"]
        vocab_size = params["vocab_size"]

        resnet = models.resnet18(weights=None)

        # --- standard ResNet18 backbone (unmodified) ---
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # project channels
        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

        # collapse height → sequence
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        # CTC classifier
        self.classifier = nn.Conv1d(out_channels, vocab_size + 1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)        # [B, 512, H, W]
        x = self.proj(x)            # [B, C, H, W]
        #x = self.pool(x)            # [B, C, 1, W]
        #x = x.squeeze(2)            # [B, C, W]
        #x = self.classifier(x)      # [B, vocab+1, W]
        #x = F.log_softmax(x, dim=1)
        return x

class ResNet18CTC_old(nn.Module):
    def __init__(self, params):
        super().__init__()
        out_channels = params["hidden_size"] 
        resnet = models.resnet18(pretrained=False)
        
        # --- modify for line images ---
        resnet.conv1.stride = (1, 1)   # keep resolution
        # remove maxpool to avoid too much downsampling
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # no maxpool
            resnet.layer1,  # 64
            resnet.layer2,  # 128
            resnet.layer3,  # 256
            resnet.layer4,  # 512
        )
        
        # collapse height → sequence
        self.proj = nn.Conv2d(512, out_channels, kernel_size=1)

        

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x

class ResNetEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        out_channels=params.get("hidden_size", 256)

        resnet = models.resnet50(pretrained=False)

        resnet.conv1.stride = (1, 1)

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # no maxpool
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x

class SeqCLREncoder2(Module):
    def __init__(self, params):
        in_channels = params["input_channels"]
        hidden = params["hidden_size"]
        super().__init__()
        self.encoder = Sequential(
            # Stage 1
            Conv2d(in_channels, 32, 3, stride=2, padding=1),  # H/2 W/2
            InstanceNorm2d(32, affine=True),
            ReLU(),

            # Stage 2
            Conv2d(32, 64, 3, stride=2, padding=1),           # H/4 W/4
            InstanceNorm2d(64, affine=True),
            ReLU(),

            # Stage 3 (keep width!)
            Conv2d(64, 128, 3, stride=(2,1), padding=1),      # H/8 W/4
            InstanceNorm2d(128, affine=True),
            ReLU(),

            # Deeper context
            Conv2d(128, hidden, 3, padding=1),
            InstanceNorm2d(hidden, affine=True),
            ReLU(),

            Conv2d(hidden, hidden, (3,5), padding=(1,2)),
            InstanceNorm2d(hidden, affine=True),
            ReLU(),

            Conv2d(hidden, hidden, (3,5), padding=(1,2)),
            InstanceNorm2d(hidden, affine=True),
            ReLU(),
        )
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.encoder(x)  # (B, C, H', W')
        return x


class SeqCLREncoder(Module):
    def __init__(self, params):
        in_channels = params["input_channels"]
        hidden = params["hidden_size"]
        super().__init__()
        self.encoder = Sequential(
            # (H, W)
            Conv2d(in_channels, 32, 3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d((2, 2)),   # H/2, W/2

            Conv2d(32, 64, 3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d((2, 2)),   # H/4, W/4

            Conv2d(64, 128, 3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d((2, 1)),   # H/8, W/4  (important: keep width!)

            Conv2d(128, hidden, 3, padding=1),
            BatchNorm2d(hidden),
            ReLU(),
        )
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.encoder(x)  # (B, C, H', W')
        return x


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)

class FCN_Encoder_Tiny(Module):
    def __init__(self, params):
        super().__init__()
        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 4,  stride=(1, 1), dropout=self.dropout),
            ConvBlock(4, 8,  stride=(2, 2), dropout=self.dropout),
            ConvBlock(8, 16, stride=(2, 2), dropout=self.dropout),
            ConvBlock(16, 32, stride=(1, 2), dropout=self.dropout),
            ConvBlock(32, 32, stride=(1, 1), dropout=self.dropout),
            ConvBlock(32, 32, stride=(1, 1), dropout=self.dropout),
        ])

        self.blocks = ModuleList([
            DSCBlock(32, 32, stride=(1, 1), dropout=self.dropout),
            DSCBlock(32, 32, stride=(1, 1), dropout=self.dropout),
            DSCBlock(32, 32, stride=(1, 1), dropout=self.dropout),
            DSCBlock(32, 64, stride=(1, 1), dropout=self.dropout),
        ])
    def forward(self, x):
        #with torch.no_grad():
        #    assert torch.isfinite(x).all(), f"Input has non-finite values: min {x.min()} max {x.max()}"
        #    print("x stats:", float(x.min()), float(x.mean()), float(x.max()))
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x

class FCN_Encoder_Small(Module):
    def __init__(self, params):
        super(FCN_Encoder_Small, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 8, stride=(1, 1), dropout=self.dropout),
            ConvBlock(8, 16, stride=(2, 2), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(1, 2), dropout=self.dropout),
            ConvBlock(64, 64, stride=(1, 1), dropout=self.dropout),
            ConvBlock(64, 64, stride=(1, 1), dropout=self.dropout),
        ])

        self.blocks = ModuleList([
            DSCBlock(64, 64, stride=(1, 1), dropout=self.dropout),
            DSCBlock(64, 64, stride=(1, 1), dropout=self.dropout),
            DSCBlock(64, 64, stride=(1, 1), dropout=self.dropout),
            DSCBlock(64, 128, stride=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        #with torch.no_grad():
        #    assert torch.isfinite(x).all(), f"Input has non-finite values: min {x.min()} max {x.max()}"
        #    print("x stats:", float(x.min()), float(x.mean()), float(x.max()))
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class FCN_Encoder(Module):
    def __init__(self, params):
        super(FCN_Encoder, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, stride=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        # x.shape=BxCxHxW
        #with torch.no_grad():
        #    assert torch.isfinite(x).all(), f"Input has non-finite values: min {x.min()} max {x.max()}"
        #    print("x stats:", float(x.min()), float(x.mean()), float(x.max()))
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(Module):

    def __init__(self, in_, out_, stride=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x
