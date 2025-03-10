from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.convnext import LayerNorm2d
from vit_pytorch.vit_for_small_dataset import ViT



class ViTSiemeseNetwork(nn.Module):
    """
    """
    def __init__(self, weights=None) -> None:
        super().__init__()

        self.vit = torchvision.models.vit_b_16(weights=weights)

        self.vit_encoder_out_dim = self.vit.heads[0].in_features # 768

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit_encoder_out_dim * 2),
            nn.Linear(self.vit_encoder_out_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )
        
        if weights is None:
            self.vit.apply(self.init_weights)
            self.classifier.apply(self.init_weights)

        self.sigmoid = nn.Sigmoid()


    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward_per_input(self, x):
        # source code:
        # https://github.com/pytorch/vision/blob/29418e34a94e2c43f861a321265f7f21035e7b19/torchvision/models/vision_transformer.py#L289C26-L289C26
        
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        b = x.shape[0]

        # Expand the class token to the full batch
        batch_cls_token = self.vit.class_token.expand(b, -1, -1)
        x = torch.cat([batch_cls_token, x], dim=1)

        x = self.vit.encoder(x)

        x = x[:, 0]

        return x
    

    def forward(self, input1, input2):
        output1 = self.forward_per_input(input1)
        output2 = self.forward_per_input(input2)

        output = torch.cat((output1, output2), 1) # shape (batch_size, 2x768)

        output = self.classifier(output)

        output = self.sigmoid(output)
        
        return output
    

class SLViTSiemeseNetwork(nn.Module):
    """
    Paper: 
    """
    def __init__(self, weights=None) -> None:
        super().__init__()

        self.vit = ViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=2 * 192,
            dim_head=192 // 12,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.vit_dim = self.vit.mlp_head[1].in_features # 192

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vit_dim * 2),
            nn.Linear(self.vit_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)
        )
        
        self.vit.apply(self.init_weights)
        self.classifier.apply(self.init_weights)

        self.sigmoid = nn.Sigmoid()


    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


    def forward_per_input(self, x):
        x = self.vit.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.vit.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit.pos_embedding[:, :(n + 1)]
        x = self.vit.dropout(x)

        x = self.vit.transformer(x)

        x = x[:, 0]

        return x
    

    def forward(self, input1, input2):
        output1 = self.forward_per_input(input1)
        output2 = self.forward_per_input(input2)

        output = torch.cat((output1, output2), 1) # shape (batch_size, 2x192)

        output = self.classifier(output)

        output = self.sigmoid(output)
        
        return output



def conv1x1(
        in_features: int,
        out_features: int,
        stride: int = 1
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False)


def conv3x3(
        in_features: int,
        out_features: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_features,
        out_features,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_features, out_features, stride)
        self.bn1 = norm_layer(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = norm_layer(out_features)

        if stride != 1:
            self.downsample = nn.Sequential(
                conv1x1(in_features, 512, stride),
                norm_layer(512),
            )
        else:
            self.downsample = None

        self.stride = stride


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNetSiameseNetwork(nn.Module):
    """
    """
    def __init__(self, weights=None, input_channels=3):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=weights)

        if input_channels == 1:
            self.resnet.conv1 = nn.Conv2d(
                1,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False
            )
        
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.encoder = torch.nn.Sequential(*(list(self.resnet.children())[:-3]))

        self.conv_block = torch.nn.Sequential(
            BasicBlock(512, 512, 2),
            BasicBlock(512, 512),
            # BasicBlock(512, 512)
        )

        self.avgpool = self.resnet.avgpool

        # add linear layers to compare between the features of the two images
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        if weights is None:
            self.resnet.apply(self.init_weights)
            self.classifier.apply(self.init_weights)

        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_per_input(self, x):
        # output = self.resnet(x) # shape (batch_size, 512, 1, 1) 
        # output = output.view(output.size()[0], -1) # shape (batch_size, 512)
        output = self.encoder(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_per_input(input1)
        output2 = self.forward_per_input(input2)

        output = torch.cat((output1, output2), 1) # shape (batch_size, 2x512)

        output = self.conv_block(output)

        output = self.avgpool(output)

        output = output.view(output.size()[0], -1)

        output = self.classifier(output)

        output = self.sigmoid(output)
        
        return output


class CLRSiameseNetwork(nn.Module):
    def __init__(self, weights=None, out_features=128) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=weights)
        self.fc_in_features = self.resnet.fc.in_features

        self.encoder = nn.Sequential(
            *list(self.resnet.children())[:-1],
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, self.fc_in_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_in_features, out_features)
        )

    def forward_per_input(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        out = F.normalize(out)
        return out
    

    def forward(self, input1, input2):
        output1 = self.forward_per_input(input1)
        output2 = self.forward_per_input(input2)

        return (output1, output2)



