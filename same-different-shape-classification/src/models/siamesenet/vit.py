import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from vit_pytorch import ViT
from vit_pytorch.vit_for_small_dataset import ViT as SLViT
from einops import repeat


class ViTB16SiameseNetwork(nn.Module):
    """
    """
    def __init__(
        self,
        weights: ViT_B_16_Weights | None = None
    ) -> None:
        super().__init__()

        self.vit = vit_b_16(weights=weights)

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.vit.heads[0].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()


    def forward_once(self, input: torch.Tensor) -> torch.Tensor:
        """
        """
        # Reshape and permute the input tensor
        output = self.vit._process_input(input)
        n = output.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        output = torch.cat([batch_class_token, output], dim=1)

        output = self.vit.encoder(output)

        # Classifier "token"
        output = output[:, 0]

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

        output = self.classifier(output)
        output = self.sigmoid(output)
        
        return output
    


class ViTSiemeseNetwork(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()

        self.vit = ViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            dim=192,
            depth=12,
            heads=12,
            mlp_dim=4 * 192,
            dim_head=192 // 12,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.vit.mlp_head.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

    
    def forward_once(self, input: torch.Tensor) -> torch.Tensor:
        """
        """
        x = self.vit.to_patch_embedding(input)
        b, n, _ = x.shape

        cls_tokens = repeat(self.vit.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit.pos_embedding[:, :(n + 1)]
        x = self.vit.dropout(x)

        x = self.vit.transformer(x)

        x = x[:, 0]

        x = self.vit.to_latent(x)

        return x
    

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

        output = self.classifier(output)
        output = self.sigmoid(output)
        
        return output

    


class SLViTSiemeseNetwork(nn.Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()

        self.vit = SLViT(
            image_size=128,
            patch_size=16,
            num_classes=2,
            dim=192,
            depth=9,
            heads=12,
            mlp_dim=2 * 192,
            dim_head=192 // 12,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * self.vit.mlp_head[1].in_features),
            nn.Linear(2 * self.vit.mlp_head[1].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

        self.sigmoid = nn.Sigmoid()

    
    def forward_once(self, input: torch.Tensor) -> torch.Tensor:
        """
        """
        x = self.vit.to_patch_embedding(input)
        b, n, _ = x.shape

        cls_tokens = repeat(self.vit.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.vit.pos_embedding[:, :(n + 1)]
        x = self.vit.dropout(x)

        x = self.vit.transformer(x)

        x = x[:, 0]

        x = self.vit.to_latent(x)

        return x
    
    
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

        output = self.classifier(output)
        output = self.sigmoid(output)
        
        return output



