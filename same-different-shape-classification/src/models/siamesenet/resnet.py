import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
)
from torchvision.models.resnet import conv1x1, BasicBlock, Bottleneck


class ResNet18SiameseNetwork(nn.Module):
    """
    """
    def __init__(
        self,
        weights: ResNet18_Weights | None = None
    ) -> None:
        super().__init__()

        self.resnet = resnet18(weights=weights)

        self.norm_layer = nn.BatchNorm2d

        self.layer4 = nn.Sequential(
            BasicBlock(
                512,
                512,
                stride=2,
                downsample=nn.Sequential(
                    conv1x1(512, 512, 2),
                    self.norm_layer(512),
                ),
                norm_layer=self.norm_layer
            ),
            BasicBlock(512, 512, norm_layer=self.norm_layer),
        )

        self.avgpool = self.resnet.avgpool

        self.classifier = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()


    def forward_once(self, input: torch.Tensor) -> torch.Tensor:
        """
        """
        output = self.resnet.conv1(input)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)

        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)

        return output



    def forward(
        self,
        input_1: torch.Tensor,
        input_2: torch.Tensor
    ) -> torch.Tensor:
        """
        """
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)
        output = torch.cat((output_1, output_2), 1)

        output = self.layer4(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)

        output = self.classifier(output)
        output = self.sigmoid(output)
        
        return output
    


class ResNet34SiameseNetwork(ResNet18SiameseNetwork):
    """
    """
    def __init__(
        self,
        weights: ResNet34_Weights | None = None
    ):
        super().__init__(weights)

        self.resnet = resnet34(weights=weights)

        self.layer4 = nn.Sequential(
            BasicBlock(
                512,
                512,
                stride=2,
                downsample=nn.Sequential(
                    conv1x1(512, 512, 2),
                    self.norm_layer(512),
                ),
                norm_layer=self.norm_layer
            ),
            BasicBlock(512, 512, norm_layer=self.norm_layer),
            BasicBlock(512, 512, norm_layer=self.norm_layer),
        )



class ResNet50SiameseNetwork(ResNet18SiameseNetwork):
    """
    """
    def __init__(
        self,
        weights: ResNet50_Weights | None = None
    ) -> None:
        super().__init__(weights)

        self.resnet = resnet50(weights=weights)

        self.layer4 = nn.Sequential(
            Bottleneck(
                2048,
                512,
                stride=2,
                downsample=nn.Sequential(
                    conv1x1(2048, 2048, 2),
                    self.norm_layer(2048),
                ),
                norm_layer=self.norm_layer
            ),
            Bottleneck(2048, 512, norm_layer=self.norm_layer),
            Bottleneck(2048, 512, norm_layer=self.norm_layer),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )



class ResNetContrastiveLearningNetwork(nn.Module):
    """
    """
    def __init__(
        self,
        base_model: str = "resnet50",
        weights: str | None = None,
        projection_dim: int = 128
    ) -> None:
        super().__init__()

        if base_model == "resnet18":
            self.resnet = resnet18(weights=ResNet18_Weights if weights == "pretrained" else None)

        elif base_model == "resnet34":
            self.resnet = resnet34(weights=ResNet34_Weights if weights == "pretrained" else None)

        elif base_model == "resnet50":
            self.resnet = resnet50(weights=ResNet50_Weights if weights == "pretrained" else None)


        self.projection_head = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.resnet.fc.in_features, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.resnet.fc.in_features, projection_dim, bias=False)
        )


    def forward_once(self, input: torch.Tensor) -> torch.Tensor:
        """
        """
        output = self.resnet.conv1(input)
        output = self.resnet.bn1(output)
        output = self.resnet.relu(output)
        output = self.resnet.maxpool(output)

        output = self.resnet.layer1(output)
        output = self.resnet.layer2(output)
        output = self.resnet.layer3(output)
        output = self.resnet.layer4(output)

        output = self.resnet.avgpool(output)
        output = torch.flatten(output, 1)

        return output
    

    def forward(
        self,
        input_1: torch.Tensor,
        input_2: torch.Tensor
    ) -> torch.Tensor:
        """
        """
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)

        output_1 = self.projection_head(output_1)
        output_2 = self.projection_head(output_2)

        return (output_1, output_2)
    


class ClassificationLinearEvaluator(nn.Module):
    """
    """
    def __init__(
        self,
        base_model: nn.Module,
        ckpt_path: str | None = None,
        num_classes: int = 2
    ) -> None:
        super().__init__()

        self.base_model = base_model

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.base_model.load_state_dict(ckpt["model"])

            for param in self.base_model.parameters():
                param.requires_grad = False

        self.num_classes = num_classes if num_classes > 2 else 1

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.base_model.resnet.fc.in_features, self.num_classes),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        input_1: torch.Tensor,
        input_2: torch.Tensor
    ) -> torch.Tensor:
        """
        """
        output_1 = self.base_model.forward_once(input_1)
        output_2 = self.base_model.forward_once(input_2)
        output = torch.cat((output_1, output_2), 1)

        output = self.classifier(output)
        output = self.sigmoid(output)
        
        return output






        




