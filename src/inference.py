import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io


class ResNet50Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    model = ResNet50Classifier().to(device)
    try:
        # weights_only=False: loading full state_dict. Only load trusted checkpoints.
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=False)
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Attention: Impossible de charger les poids ({e}). Modèle vide.")
        return model


def predict_image(model: nn.Module, image_bytes: bytes, device: torch.device) -> dict:
    # Transformation standard pour ResNet
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        prob = torch.sigmoid(logits).item()

    prediction = "Métastase" if prob >= 0.5 else "Sain"

    return {
        "diagnostic": prediction,
        "probabilite": round(prob, 4),
        "risque_pourcentage": f"{prob * 100:.2f}%",
    }
