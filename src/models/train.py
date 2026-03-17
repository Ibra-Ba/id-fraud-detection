"""
EfficientNet-B0 binary classifier (genuine vs fraud).
Backbone from timm, pretrained on ImageNet.
Optimisé CPU — pas de GPU requis.
"""

import torch
import torch.nn as nn
import timm


class FraudClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuné pour classification binaire genuine/fraud.

    Stratégie 2 phases :
      - Phase 1 : backbone gelé, entraînement de la tête seulement
      - Phase 2 : backbone dégelé, fine-tuning complet (lr réduit)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # retire la tête originale
        )
        in_features = self.backbone.num_features  # 1280 pour B0

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Phase 1 — gèle le backbone, entraîne la tête uniquement."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone gelé — entraînement tête uniquement")

    def unfreeze_backbone(self) -> None:
        """Phase 2 — dégèle tout pour fine-tuning complet."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] Backbone dégelé — fine-tuning complet")

    def count_parameters(self) -> dict:
        """Retourne le nombre de paramètres trainables / total."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
