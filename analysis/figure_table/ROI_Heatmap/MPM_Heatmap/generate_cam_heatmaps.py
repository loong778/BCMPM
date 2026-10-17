import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
Image.MAX_IMAGE_PIXELS = None  # See README.md for project setup and required files.


# ==============================
# See README.md for project setup and required files.
# ==============================
class Config:
    # See README.md for project setup and required files.
    # Required input: see README.md for the required file and directory layout.
    INPUT_DIR = "./xxx/xx"  # See README.md for project setup and required files.
    # Example output path; generated files are written here.
    OUTPUT_DIR = "./xxx/xx"  # Example output directory for generated MPM heatmaps.
    # Required input: see README.md for the required file and directory layout.
    MODEL_PATH = "./xxx/xx/model_checkpoint.pth"  # Required input: trained heatmap model checkpoint.

    # See README.md for project setup and required files.
    NUM_CLASSES = 8
    IMAGE_SIZE = 456
    BACKBONE = 'efficientnet'
    EFFICIENTNET_VERSION = 'efficientnet-b5'

    # See README.md for project setup and required files.
    HEATMAP_ALPHA = 0.6  # See README.md for project setup and required files.
    COLORMAP = cv2.COLORMAP_JET  # See README.md for project setup and required files.

    # See README.md for project setup and required files.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


# ==============================
# See README.md for project setup and required files.
# ==============================
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_attention(x)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        return x


# ==============================
# See README.md for project setup and required files.
# ==============================
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

        # See README.md for project setup and required files.
        if config.BACKBONE == 'efficientnet':
            from efficientnet_pytorch import EfficientNet
            self.backbone = EfficientNet.from_name(config.EFFICIENTNET_VERSION)
            in_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            self.feature_extractor = self.backbone.extract_features
        else:
            self.backbone = models.resnet50(pretrained=False)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.feature_extractor = self.backbone

        # See README.md for project setup and required files.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.7)
        )

        # See README.md for project setup and required files.
        self.attention = CBAM(in_features)

        # See README.md for project setup and required files.
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.56),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.42),
            nn.Linear(128, num_classes)
        )

        # See README.md for project setup and required files.
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            if self.activations.requires_grad:
                self.activations.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # See README.md for project setup and required files.
        if hasattr(self.backbone, '_conv_head'):  # EfficientNet
            target_layer = self.backbone._conv_head
        elif hasattr(self.backbone, 'layer4'):  # ResNet
            target_layer = self.backbone.layer4[-1].conv3
        else:
            target_layer = list(self.backbone.children())[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        if x.size(2) != config.IMAGE_SIZE or x.size(3) != config.IMAGE_SIZE:
            x = F.interpolate(x, size=(config.IMAGE_SIZE, config.IMAGE_SIZE), mode='bilinear')

        features = self.feature_extractor(x)
        features = self.attention(features)

        if len(features.shape) == 4:
            global_feat = self.global_pool(features)
            global_feat = global_feat.view(global_feat.size(0), -1)
        else:
            global_feat = features

        return self.classifier(global_feat)

    def generate_cam(self, input_tensor, target_class=None):
        """See README.md for the script purpose and required inputs."""
        self.eval()
        input_tensor = input_tensor.unsqueeze(0).to(config.DEVICE)
        input_tensor.requires_grad = True

        # See README.md for project setup and required files.
        logits = self.forward(input_tensor)
        probs = torch.sigmoid(logits)[0]

        # See README.md for project setup and required files.
        if target_class is None:
            target_class = torch.argmax(probs).item()

        # See README.md for project setup and required files.
        self.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        # See README.md for project setup and required files.
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # See README.md for project setup and required files.
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam

        return cam, probs.detach().cpu().numpy(), target_class


# ==============================
# See README.md for project setup and required files.
# ==============================
def preprocess_image(image_path):
    """See README.md for the script purpose and required inputs."""
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image), np.array(image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))


# ==============================
# See README.md for project setup and required files.
# ==============================
def generate_and_save_heatmap(model, image_path, output_path):
    """See README.md for the script purpose and required inputs."""
    try:
        # See README.md for project setup and required files.
        input_tensor, original_image = preprocess_image(image_path)

        # See README.md for project setup and required files.
        cam, probs, target_class = model.generate_cam(input_tensor)

        # See README.md for project setup and required files.
        cam_resized = cv2.resize(cam, (config.IMAGE_SIZE, config.IMAGE_SIZE))

        # See README.md for project setup and required files.
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), config.COLORMAP)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # See README.md for project setup and required files.
        overlay = cv2.addWeighted(original_image, 1 - config.HEATMAP_ALPHA,
                                  heatmap, config.HEATMAP_ALPHA, 0)

        # See README.md for project setup and required files.
        plt.figure(figsize=(config.IMAGE_SIZE / 100, config.IMAGE_SIZE / 100), dpi=100)
        plt.imshow(overlay)
        plt.axis('off')  # See README.md for project setup and required files.

        # See README.md for project setup and required files.
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close()

        print(f"✅ Status: {os.path.basename(output_path)}")
        print(f"   Status: C{target_class + 1}, Status: {probs[target_class]:.3f}")

    except Exception as e:
        print(f"❌ Status {os.path.basename(image_path)}: {str(e)}")


# ==============================
# See README.md for project setup and required files.
# ==============================
def main():
    print("=" * 60)
    print("Processing...")
    print("=" * 60)

    # See README.md for project setup and required files.
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(config.INPUT_DIR):
        print(f"Status: Status {config.INPUT_DIR}")
        return

    if not os.path.exists(config.MODEL_PATH):
        print(f"Status: Status {config.MODEL_PATH}")
        return

    # See README.md for project setup and required files.
    image_files = [f for f in os.listdir(config.INPUT_DIR)
                   if f.lower().endswith('.tif')]

    if not image_files:
        print(f"Status: Status {config.INPUT_DIR} Status.tifStatus")
        return

    print(f"Status {len(image_files)} Status.tifStatus")
    print(f"Status: {config.DEVICE}")
    print(f"Status: {config.INPUT_DIR}")
    print(f"Status: {config.OUTPUT_DIR}")
    print("-" * 60)

    # See README.md for project setup and required files.
    try:
        model = MultiLabelModel(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Processing...")
    except Exception as e:
        print(f"Status: {str(e)}")
        return

    # See README.md for project setup and required files.
    print("Processing...")
    print("-" * 60)

    for i, filename in enumerate(image_files):
        image_path = os.path.join(config.INPUT_DIR, filename)
        output_filename = f"cam_overlay_{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(config.OUTPUT_DIR, output_filename)

        generate_and_save_heatmap(model, image_path, output_path)

    print("-" * 60)
    print(f"Status {len(image_files)} Status")
    print(f"Status: {os.path.abspath(config.OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
