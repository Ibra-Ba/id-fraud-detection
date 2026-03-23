import timm
import torch
import torch.nn as nn


class FraudClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features

        # IMPORTANT : On utilise 'classifier' pour correspondre aux tests
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# --- FONCTIONS DE PONT (WRAPPERS) À AJOUTER ICI ---


def build_model(num_classes: int = 2, pretrained: bool = True) -> FraudClassifier:
    """Instancie le modèle pour les tests et l'entraînement."""
    return FraudClassifier(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model: FraudClassifier) -> None:
    """Appelle la méthode de freeze sur l'instance."""
    model.freeze_backbone()


def unfreeze_backbone(model: FraudClassifier) -> None:
    """Appelle la méthode d'unfreeze sur l'instance."""
    model.unfreeze_backbone()
