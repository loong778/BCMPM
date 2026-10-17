"""See README.md for the script purpose and required inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import json
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import warnings
import math
import logging
from functools import lru_cache

warnings.filterwarnings('ignore')

# See README.md for project setup and required files.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class ROIEmbedding:
    """See README.md for the script purpose and required inputs."""
    patient_id: str
    tacs_type: int
    he_feature: np.ndarray  # See README.md for project setup and required files.
    mpm_feature: np.ndarray  # See README.md for project setup and required files.
    fused_feature: np.ndarray  # See README.md for project setup and required files.
    he_path: str
    mpm_path: str
    filename: str
    split: str  # See README.md for project setup and required files.
    he_patch_features: Optional[np.ndarray] = None  # See README.md for project setup and required files.
    mpm_patch_features: Optional[np.ndarray] = None  # See README.md for project setup and required files.


class PatchExtractor:
    """See README.md for the script purpose and required inputs."""

    def __init__(self, patch_size: int = 224, stride: int = 224, min_patch_size: int = 224):
        """See README.md for the script purpose and required inputs."""
        self.patch_size = patch_size
        self.stride = stride
        self.min_patch_size = min_patch_size

        # See README.md for project setup and required files.
        self.patch_transform = T.Compose([
            T.Resize((min_patch_size, min_patch_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract_patches(self, image: Image.Image) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
        """See README.md for the script purpose and required inputs."""
        # See README.md for project setup and required files.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # See README.md for project setup and required files.
        width, height = image.size

        # See README.md for project setup and required files.
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size

        patches = []
        positions = []

        # See README.md for project setup and required files.
        for y in range(num_patches_y):
            for x in range(num_patches_x):
                # See README.md for project setup and required files.
                start_x = x * self.patch_size
                start_y = y * self.patch_size
                end_x = start_x + self.patch_size
                end_y = start_y + self.patch_size

                # See README.md for project setup and required files.
                patch = image.crop((start_x, start_y, end_x, end_y))

                # See README.md for project setup and required files.
                patch_tensor = self.patch_transform(patch)

                patches.append(patch_tensor)
                positions.append((start_x, start_y))

        # See README.md for project setup and required files.
        if not patches:
            # See README.md for project setup and required files.
            adjusted_patch_size = min(width, height, self.patch_size)
            if adjusted_patch_size < self.min_patch_size:
                adjusted_patch_size = self.min_patch_size

            # See README.md for project setup and required files.
            center_x = width // 2
            center_y = height // 2
            half_size = adjusted_patch_size // 2

            left = max(0, center_x - half_size)
            upper = max(0, center_y - half_size)
            right = min(width, center_x + half_size)
            lower = min(height, center_y + half_size)

            patch = image.crop((left, upper, right, lower))

            # See README.md for project setup and required files.
            if patch.size[0] < self.min_patch_size or patch.size[1] < self.min_patch_size:
                patch = patch.resize((self.min_patch_size, self.min_patch_size))

            patch_tensor = self.patch_transform(patch)
            patches.append(patch_tensor)
            positions.append((left, upper))

        return patches, positions

    def extract_patches_adaptive(self, image: Image.Image, target_num_patches: int = 16) -> Tuple[
        List[torch.Tensor], List[Tuple[int, int]]]:
        """See README.md for the script purpose and required inputs."""
        # See README.md for project setup and required files.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size

        # See README.md for project setup and required files.
        grid_size = int(math.sqrt(target_num_patches))

        # See README.md for project setup and required files.
        patch_width = (width // grid_size) // 224 * 224
        patch_height = (height // grid_size) // 224 * 224

        # See README.md for project setup and required files.
        patch_width = max(224, patch_width)
        patch_height = max(224, patch_height)

        patches = []
        positions = []

        for i in range(grid_size):
            for j in range(grid_size):
                x = j * patch_width
                y = i * patch_height

                # See README.md for project setup and required files.
                if x + 224 > width or y + 224 > height:
                    continue

                # See README.md for project setup and required files.
                patch = image.crop((x, y, x + 224, y + 224))

                # See README.md for project setup and required files.
                patch_tensor = self.patch_transform(patch)

                patches.append(patch_tensor)
                positions.append((x, y))

                if len(patches) >= target_num_patches:
                    break
            if len(patches) >= target_num_patches:
                break

        return patches, positions


def get_norm_constants(which_img_norm: str = 'imagenet'):
    """See README.md for the script purpose and required inputs."""
    constants_zoo = {
        'imagenet': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'ctranspath': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
        'openai_clip': {'mean': (0.48145466, 0.4578275, 0.40821073), 'std': (0.26862954, 0.26130258, 0.27577711)},
        'uniform': {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    }
    constants = constants_zoo[which_img_norm]
    return constants.get('mean'), constants.get('std')


def get_eval_transforms(
        which_img_norm: str = 'imagenet',
        img_resize: int = 224,
        center_crop: bool = False
):
    """See README.md for the script purpose and required inputs."""

    eval_transform = []

    if img_resize > 0:
        eval_transform.append(T.Resize(img_resize))

        if center_crop:
            eval_transform.append(T.CenterCrop(img_resize))

    mean, std = get_norm_constants(which_img_norm)

    eval_transform.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    eval_transform = T.Compose(eval_transform)
    return eval_transform


class HEFeatureExtractor(nn.Module):
    """See README.md for the script purpose and required inputs."""

    def __init__(self, output_dim: int = 1024, device: torch.device = None):
        super().__init__()
        self.output_dim = output_dim
        self.device = device

        print("=" * 50)
        print("Processing...")
        print("=" * 50)

        try:
            # See README.md for project setup and required files.
            try:
                from huggingface_hub import login, hf_hub_download
                print("Processing...")
            except ImportError as e:
                print(f"❌ huggingface_hubStatus: {e}")
                print("Processing...")
                raise ImportError("Processing...")

            # See README.md for project setup and required files.
            uni_project_paths = [
                "./uni",  # See README.md for project setup and required files.
                "./xxx/xx",
                "/mnt/e/code/CLAM-master/uni",
                os.path.expanduser("~/code/CLAM-master/uni"),
                os.path.dirname(__file__),  # See README.md for project setup and required files.
                os.path.join(os.path.dirname(__file__), "uni"),  # See README.md for project setup and required files.
                "."  # See README.md for project setup and required files.
            ]

            for uni_path in uni_project_paths:
                if os.path.exists(uni_path):
                    if uni_path not in sys.path:
                        sys.path.append(uni_path)
                        print(f"✓ StatusUNIStatus: {uni_path}")

                    # See README.md for project setup and required files.
                    get_encoder_path = os.path.join(uni_path, "get_encoder", "models")
                    if os.path.exists(get_encoder_path):
                        if get_encoder_path not in sys.path:
                            sys.path.append(get_encoder_path)
                        print(f"✓ Statusget_encoderStatus: {get_encoder_path}")
                        break

            # See README.md for project setup and required files.
            uni_weight_paths = [
                # See README.md for project setup and required files.
                "./uni_weights/pytorch_model.bin",
                # See README.md for project setup and required files.
                "./xxx/xx",
                os.path.expanduser("~/.cache/uni/pytorch_model.bin"),
                "pretrained/uni/pytorch_model.bin",
                "./pytorch_model.bin",  # See README.md for project setup and required files.
                "uni/pytorch_model.bin",  # See README.md for project setup and required files.
                # See README.md for project setup and required files.
                os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]),
                             '../../assets/ckpts/uni/pytorch_model.bin')
            ]

            model = None
            preprocess = None
            ckpt_loaded = False

            # See README.md for project setup and required files.
            for weight_path in uni_weight_paths:
                try:
                    ckpt_path = weight_path

                    # See README.md for project setup and required files.
                    if os.path.exists(ckpt_path):
                        print(f"StatusUNIStatus: {ckpt_path}")
                        file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
                        print(f"Status: {file_size:.2f} MB")
                    else:
                        print(f"Status: {ckpt_path}")
                        # See README.md for project setup and required files.
                        ckpt_dir = os.path.dirname(ckpt_path)
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir, exist_ok=True)

                        print(f"StatusHugging Face HubStatusUNIStatus: {ckpt_dir}")
                        try:
                            # See README.md for project setup and required files.
                            hf_hub_download(
                                'MahmoodLab/UNI',
                                filename="pytorch_model.bin",
                                local_dir=ckpt_dir,
                                force_download=True
                            )
                            print("Processing...")
                        except Exception as download_e:
                            print(f"Hugging Face HubStatus: {download_e}")
                            continue

                    # See README.md for project setup and required files.
                    if os.path.exists(ckpt_path):
                        # See README.md for project setup and required files.
                        print("Processing...")

                        # See README.md for project setup and required files.
                        uni_kwargs = {
                            'model_name': 'vit_large_patch16_224',
                            'img_size': 224,
                            'patch_size': 16,
                            'init_values': 1e-5,
                            'num_classes': 0,  # See README.md for project setup and required files.
                            'dynamic_img_size': True
                        }

                        # See README.md for project setup and required files.
                        model = timm.create_model(**uni_kwargs)
                        print(f"✓ StatusUNIStatus: vit_large_patch16_224")

                        # See README.md for project setup and required files.
                        state_dict = torch.load(ckpt_path, map_location="cpu")
                        print(f"✓ Status: {len(state_dict)}")

                        # See README.md for project setup and required files.
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
                        print(f"Status: {missing_keys}")
                        print(f"Status: {unexpected_keys}")

                        # See README.md for project setup and required files.
                        preprocess = get_eval_transforms(
                            which_img_norm='imagenet',
                            img_resize=224,
                            center_crop=True
                        )

                        print("Processing...")
                        ckpt_loaded = True
                        break

                except Exception as e:
                    print(f"Status {ckpt_path} Status: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if not ckpt_loaded:
                print("Processing...")
                try:
                    # See README.md for project setup and required files.
                    uni_kwargs = {
                        'model_name': 'vit_large_patch16_224',
                        'img_size': 224,
                        'patch_size': 16,
                        'init_values': 1e-5,
                        'num_classes': 0,
                        'dynamic_img_size': True
                    }
                    model = timm.create_model(**uni_kwargs)
                    preprocess = get_eval_transforms(
                        which_img_norm='imagenet',
                        img_resize=224,
                        center_crop=True
                    )
                    print("Processing...")
                except Exception as e:
                    print(f"StatusUNIStatus: {e}")
                    print("Processing...")
                    raise e

            # See README.md for project setup and required files.
            model.eval()

            # See README.md for project setup and required files.
            for param in model.parameters():
                param.requires_grad = False

            # See README.md for project setup and required files.
            class UNIEncoder(nn.Module):
                def __init__(self, visual_model, preprocess):
                    super().__init__()
                    self.visual = visual_model
                    self.preprocess = preprocess

                def encode_image(self, x, proj_contrast=False, normalize=False):
                    # See README.md for project setup and required files.
                    output = self.visual(x)
                    return output

                def forward(self, x):
                    return self.encode_image(x)

            self.model = UNIEncoder(model, preprocess)

            # See README.md for project setup and required files.
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                test_output = self.model(test_input)
                # See README.md for project setup and required files.
                self.feature_dim = test_output.shape[-1]

            print(f"✓ StatusUNIStatusHEStatus: {self.feature_dim}")

        except ImportError as e:
            print(f"UNIStatus: {e}, StatusResNet50")
            self._init_resnet_fallback()
        except Exception as e:
            print(f"UNIStatus: {e}, StatusResNet50")
            self._init_resnet_fallback()

    def _init_resnet_fallback(self):
        """See README.md for the script purpose and required inputs."""
        print("Processing...")
        self.model = models.resnet50(pretrained=True)
        # See README.md for project setup and required files.
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # See README.md for project setup and required files.
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_dim = 2048
        self.preprocess = None
        print("Processing...")

    def extract_patch_features(self, patches: List[torch.Tensor], batch_size: int = 64) -> torch.Tensor:
        """See README.md for the script purpose and required inputs."""
        if not patches:
            return torch.empty((0, self.feature_dim), device=self.device)

        all_features = []

        # See README.md for project setup and required files.
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_tensor = torch.stack(batch_patches)

            # See README.md for project setup and required files.
            if self.device is not None:
                batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                if hasattr(self.model, 'encode_image'):
                    # See README.md for project setup and required files.
                    batch_features = self.model.encode_image(batch_tensor)
                else:
                    # See README.md for project setup and required files.
                    batch_features = self.model(batch_tensor)
                    if batch_features.dim() == 4:
                        batch_features = F.adaptive_avg_pool2d(batch_features, (1, 1))
                    batch_features = batch_features.view(batch_features.size(0), -1)

            all_features.append(batch_features.cpu())

        # See README.md for project setup and required files.
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.empty((0, self.feature_dim), device=self.device)

    def aggregate_patch_features(self, patch_features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """See README.md for the script purpose and required inputs."""
        if patch_features.size(0) == 0:
            return torch.zeros(self.feature_dim, device=self.device)

        if method == 'mean':
            # See README.md for project setup and required files.
            roi_feature = patch_features.mean(dim=0)
        elif method == 'max':
            # See README.md for project setup and required files.
            roi_feature, _ = patch_features.max(dim=0)
        elif method == 'attention':
            # See README.md for project setup and required files.
            # See README.md for project setup and required files.
            roi_feature = patch_features.mean(dim=0)
        else:
            raise ValueError(f"Status: {method}")

        return roi_feature


class MPMFeatureExtractor(nn.Module):
    """See README.md for the script purpose and required inputs."""

    def __init__(self, output_dim: int = 1024, device: torch.device = None):  # See README.md for project setup and required files.
        super().__init__()
        self.output_dim = output_dim
        self.device = device

        print("=" * 50)
        print("Processing...")
        print(f"✓ Status: {output_dim} (StatusHEStatusHEStatus1024)")
        print("=" * 50)

        # See README.md for project setup and required files.
        local_weight_paths = [
            "./xxx/xx",
            "./xxx/xx",
            os.path.expanduser("~/.cache/torch/hub/checkpoints/efficientnet-b3-5fb5a3c3.pth"),
            os.path.expanduser("~/.cache/torch/hub/checkpoints/efficientnet_b3_ra2-cf984f9c.pth"),
            "/tmp/efficientnet-b3-5fb5a3c3.pth",
        ]

        # See README.md for project setup and required files.
        local_weight_found = None
        for weight_path in local_weight_paths:
            if os.path.exists(weight_path):
                local_weight_found = weight_path
                print(f"Status: {weight_path}")
                break

        # See README.md for project setup and required files.
        if local_weight_found:
            print(f"StatusEfficientNetStatus: {local_weight_found}")
            success, model = self._load_local_weights(local_weight_found)
            if success:
                self.model = model
                print("Processing...")
            else:
                print("Processing...")
                self.model = self._load_and_cache_official_weights()
        else:
            print("Processing...")
            self.model = self._load_and_cache_official_weights()

        # See README.md for project setup and required files.
        for param in self.model.parameters():
            param.requires_grad = False

        # See README.md for project setup and required files.
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.model(test_input)
            # See README.md for project setup and required files.
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            self.raw_feature_dim = features.shape[1]

        # See README.md for project setup and required files.
        self.projection = nn.Linear(self.raw_feature_dim, 1024)  # See README.md for project setup and required files.

        # See README.md for project setup and required files.
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # See README.md for project setup and required files.
        for param in self.projection.parameters():
            param.requires_grad = False

        # See README.md for project setup and required files.
        self.feature_dim = 1024  # See README.md for project setup and required files.

        print(f"Status: {self.raw_feature_dim} -> Status: {self.feature_dim}")
        print(f"✓ StatusEfficientNet-B3StatusMPMStatus: {self.feature_dim} (StatusHEStatus)")

    def _load_local_weights(self, weight_path: str) -> Tuple[bool, Optional[nn.Module]]:
        """See README.md for the script purpose and required inputs."""
        try:
            # See README.md for project setup and required files.
            state_dict = torch.load(weight_path, map_location='cpu')

            # See README.md for project setup and required files.
            if '_conv_stem.weight' in state_dict:
                print("Processing...")
                # See README.md for project setup and required files.
                model = timm.create_model(
                    'efficientnet_b3',
                    pretrained=False,
                    num_classes=0,  # See README.md for project setup and required files.
                    features_only=False
                )

                # See README.md for project setup and required files.
                model.load_state_dict(state_dict, strict=False)
                print("Processing...")

                return True, model

            elif 'features.0.0.weight' in state_dict or 'classifier.1.weight' in state_dict:
                print("Processing...")
                # See README.md for project setup and required files.
                model = models.efficientnet_b3(pretrained=False)

                # See README.md for project setup and required files.
                model.load_state_dict(state_dict)
                print("Processing...")

                # See README.md for project setup and required files.
                model = nn.Sequential(*list(model.children())[:-1])
                return True, model

            else:
                print("Processing...")
                return False, None

        except Exception as e:
            print(f"Status: {e}")
            return False, None

    def _load_and_cache_official_weights(self) -> nn.Module:
        """See README.md for the script purpose and required inputs."""
        # See README.md for project setup and required files.
        cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # See README.md for project setup and required files.
        cache_file = cache_dir / 'efficientnet-b3-5fb5a3c3.pth'

        # See README.md for project setup and required files.
        if cache_file.exists():
            print(f"Status: {cache_file}")
            # See README.md for project setup and required files.
            success, model = self._load_local_weights(str(cache_file))
            if success:
                return model

        # See README.md for project setup and required files.
        print("Processing...")
        try:
            # See README.md for project setup and required files.
            model = timm.create_model(
                'efficientnet_b3',
                pretrained=True,
                num_classes=0,  # See README.md for project setup and required files.
                features_only=False
            )
            print("Processing...")

            # See README.md for project setup and required files.
            try:
                torch.save(model.state_dict(), cache_file)
                print(f"Status: {cache_file}")
            except Exception as e:
                print(f"⚠ Status: {e}")

            return model

        except Exception as e:
            print(f"Status: {e}")
            print("Processing...")
            return timm.create_model(
                'efficientnet_b3',
                pretrained=False,
                num_classes=0,
                features_only=False
            )

    def extract_patch_features(self, patches: List[torch.Tensor], batch_size: int = 64) -> torch.Tensor:
        """See README.md for the script purpose and required inputs."""
        if not patches:
            return torch.empty((0, self.feature_dim), device=self.device)

        all_features = []

        # See README.md for project setup and required files.
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_tensor = torch.stack(batch_patches)

            # See README.md for project setup and required files.
            if self.device is not None:
                batch_tensor = batch_tensor.to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)
                # See README.md for project setup and required files.
                if features.dim() > 2:
                    # See README.md for project setup and required files.
                    if features.dim() == 4:
                        features = F.adaptive_avg_pool2d(features, (1, 1))
                    features = features.view(features.size(0), -1)

                # See README.md for project setup and required files.
                features = self.projection(features)

            all_features.append(features.cpu())

        # See README.md for project setup and required files.
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.empty((0, self.feature_dim), device=self.device)

    def aggregate_patch_features(self, patch_features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """See README.md for the script purpose and required inputs."""
        if patch_features.size(0) == 0:
            return torch.zeros(self.feature_dim, device=self.device)

        if method == 'mean':
            # See README.md for project setup and required files.
            roi_feature = patch_features.mean(dim=0)
        elif method == 'max':
            # See README.md for project setup and required files.
            roi_feature, _ = patch_features.max(dim=0)
        elif method == 'attention':
            # See README.md for project setup and required files.
            # See README.md for project setup and required files.
            roi_feature = patch_features.mean(dim=0)
        else:
            raise ValueError(f"Status: {method}")

        return roi_feature


class CrossAttentionFusionModule(nn.Module):
    """See README.md for the script purpose and required inputs."""

    def __init__(self,
                 he_feature_dim: int = 1024,  # See README.md for project setup and required files.
                 mpm_feature_dim: int = 1024,  # See README.md for project setup and required files.
                 output_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 device: torch.device = None):
        super().__init__()

        self.he_feature_dim = he_feature_dim
        self.mpm_feature_dim = mpm_feature_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.device = device

        print("=" * 50)
        print("Processing...")
        print(f"HEStatus: {he_feature_dim}")
        print(f"MPMStatus: {mpm_feature_dim}")
        print(f"Status: {num_heads}")
        print(f"Status: {output_dim} (Status)")
        print("=" * 50)

        # See README.md for project setup and required files.
        assert he_feature_dim == mpm_feature_dim, "Processing..."

        # See README.md for project setup and required files.
        self.proj_dim = 256  # See README.md for project setup and required files.
        self.he_proj = nn.Linear(he_feature_dim, self.proj_dim)
        self.mpm_proj = nn.Linear(mpm_feature_dim, self.proj_dim)

        # See README.md for project setup and required files.
        # See README.md for project setup and required files.
        self.he_to_mpm_attention = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # See README.md for project setup and required files.
        self.mpm_to_he_attention = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # See README.md for project setup and required files.
        self.norm1 = nn.LayerNorm(self.proj_dim)
        self.norm2 = nn.LayerNorm(self.proj_dim)

        # See README.md for project setup and required files.
        self.feed_forward = nn.Sequential(
            nn.Linear(self.proj_dim * 2, 512),  # See README.md for project setup and required files.
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)  # See README.md for project setup and required files.
        )

        # See README.md for project setup and required files.
        self.final_norm = nn.LayerNorm(output_dim)

        # See README.md for project setup and required files.
        if device is not None:
            self.to(device)

    def forward(self, he_features: torch.Tensor, mpm_features: torch.Tensor) -> torch.Tensor:
        """See README.md for the script purpose and required inputs."""
        # See README.md for project setup and required files.
        device = next(self.parameters()).device
        he_features = he_features.to(device)
        mpm_features = mpm_features.to(device)

        # See README.md for project setup and required files.
        he_proj = self.he_proj(he_features).unsqueeze(1)  # [B, 1, proj_dim]
        mpm_proj = self.mpm_proj(mpm_features).unsqueeze(1)  # [B, 1, proj_dim]

        # See README.md for project setup and required files.
        he_attended, he_attention_weights = self.he_to_mpm_attention(
            query=he_proj,
            key=mpm_proj,
            value=mpm_proj
        )

        # See README.md for project setup and required files.
        he_fused = self.norm1(he_proj + he_attended)

        # See README.md for project setup and required files.
        mpm_attended, mpm_attention_weights = self.mpm_to_he_attention(
            query=mpm_proj,
            key=he_proj,
            value=he_proj
        )

        # See README.md for project setup and required files.
        mpm_fused = self.norm2(mpm_proj + mpm_attended)

        # See README.md for project setup and required files.
        combined = torch.cat([he_fused.squeeze(1), mpm_fused.squeeze(1)], dim=1)

        # See README.md for project setup and required files.
        output = self.feed_forward(combined)

        # See README.md for project setup and required files.
        output = self.final_norm(output)

        return output


class TACSAwareCrossAttentionFusion(nn.Module):
    """See README.md for the script purpose and required inputs."""

    def __init__(self,
                 he_feature_dim: int = 1024,  # See README.md for project setup and required files.
                 mpm_feature_dim: int = 1024,  # See README.md for project setup and required files.
                 tacs_emb_dim: int = 32,
                 output_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 device: torch.device = None):
        super().__init__()

        self.tacs_emb_dim = tacs_emb_dim
        self.output_dim = output_dim  # See README.md for project setup and required files.
        self.device = device

        # See README.md for project setup and required files.
        self.tacs_embedding = nn.Embedding(9, tacs_emb_dim)  # See README.md for project setup and required files.

        # See README.md for project setup and required files.
        self.cross_attention_fusion = CrossAttentionFusionModule(
            he_feature_dim=he_feature_dim,
            mpm_feature_dim=mpm_feature_dim,
            output_dim=384,  # See README.md for project setup and required files.
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )

        # See README.md for project setup and required files.
        self.tacs_fusion = nn.Sequential(
            nn.Linear(384 + tacs_emb_dim, 1024),  # See README.md for project setup and required files.
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)  # See README.md for project setup and required files.
        )

        # See README.md for project setup and required files.
        self.norm = nn.LayerNorm(output_dim)

        # See README.md for project setup and required files.
        if device is not None:
            self.to(device)

        print(f"TACSStatus: Status=384, TACSStatus={tacs_emb_dim}, Status={output_dim}")

    def forward(self, he_features: torch.Tensor, mpm_features: torch.Tensor, tacs_types: torch.Tensor) -> torch.Tensor:
        """See README.md for the script purpose and required inputs."""
        # See README.md for project setup and required files.
        device = next(self.parameters()).device

        # See README.md for project setup and required files.
        he_features = he_features.to(device)
        mpm_features = mpm_features.to(device)
        tacs_types = tacs_types.to(device)

        # See README.md for project setup and required files.
        tacs_emb = self.tacs_embedding(tacs_types)

        # See README.md for project setup and required files.
        image_fused = self.cross_attention_fusion(he_features, mpm_features)

        # See README.md for project setup and required files.
        combined = torch.cat([image_fused, tacs_emb], dim=1)

        # See README.md for project setup and required files.
        output = self.tacs_fusion(combined)

        # See README.md for project setup and required files.
        normalized_output = self.norm(output)

        return normalized_output


class MultimodalFusionModel(nn.Module):
    """See README.md for the script purpose and required inputs."""

    def __init__(self,
                 he_model_type: str = 'uni',
                 mpm_model_type: str = 'efficientnet_b3',
                 fusion_output_dim: int = 256,
                 fusion_method: str = 'cross_attention',
                 use_tacs_aware: bool = True,
                 patch_size: int = 224,
                 patch_stride: int = 224,
                 patch_aggregation: str = 'mean',
                 device: torch.device = None):
        super().__init__()

        self.fusion_output_dim = fusion_output_dim
        self.fusion_method = fusion_method
        self.use_tacs_aware = use_tacs_aware
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_aggregation = patch_aggregation
        self.device = device

        print("=" * 70)
        print("Processing...")
        print(f"  HEStatus: {he_model_type} (Statusget_encoder.pyStatusUNIStatus1024)")
        print(f"  MPMStatus: {mpm_model_type} (EfficientNet-B3Status1024Status)")
        print(f"  Status: {fusion_method}")
        print(f"  TACSStatus: {use_tacs_aware}")
        print(f"  Status: {fusion_output_dim} (Status)")
        print(f"  PatchStatus: {patch_size} (Status)")
        print(f"  PatchStatus: {patch_stride} (Status)")
        print(f"  PatchStatus: {patch_aggregation}")
        print(f"  Status: {device}")
        print("=" * 70)

        # See README.md for project setup and required files.
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            stride=patch_stride,
            min_patch_size=224
        )

        # See README.md for project setup and required files.
        self.he_extractor = HEFeatureExtractor(device=device)
        self.mpm_extractor = MPMFeatureExtractor(device=device)

        # See README.md for project setup and required files.
        he_dim = self.he_extractor.feature_dim
        mpm_dim = self.mpm_extractor.feature_dim

        print(f"Status: HE={he_dim}, MPM={mpm_dim}")

        # See README.md for project setup and required files.
        if he_dim != mpm_dim:
            print(f"Status: HEStatus({he_dim})StatusMPMStatus({mpm_dim})Status!")

        # See README.md for project setup and required files.
        if use_tacs_aware:
            # See README.md for project setup and required files.
            self.fusion_module = TACSAwareCrossAttentionFusion(
                he_feature_dim=he_dim,
                mpm_feature_dim=mpm_dim,
                tacs_emb_dim=32,
                output_dim=fusion_output_dim,
                num_heads=8,
                dropout=0.1,
                device=device
            )
        else:
            # See README.md for project setup and required files.
            self.fusion_module = CrossAttentionFusionModule(
                he_feature_dim=he_dim,
                mpm_feature_dim=mpm_dim,
                output_dim=fusion_output_dim,
                num_heads=8,
                dropout=0.1,
                device=device
            )

        print(f"✓ Status: {fusion_output_dim}")

    def process_roi(self,
                    he_image: Image.Image,
                    mpm_image: Image.Image,
                    tacs_type: int,
                    patch_batch_size: int = 256) -> Dict[str, torch.Tensor]:
        """See README.md for the script purpose and required inputs."""

        # See README.md for project setup and required files.
        he_patches, he_positions = self.patch_extractor.extract_patches(he_image)
        mpm_patches, mpm_positions = self.patch_extractor.extract_patches(mpm_image)

        print(f"  HEStatus {len(he_patches)} Statuspatches")
        print(f"  MPMStatus {len(mpm_patches)} Statuspatches")

        # See README.md for project setup and required files.
        he_patch_features = self.he_extractor.extract_patch_features(he_patches, batch_size=patch_batch_size)
        mpm_patch_features = self.mpm_extractor.extract_patch_features(mpm_patches, batch_size=patch_batch_size)

        # See README.md for project setup and required files.
        he_roi_feature = self.he_extractor.aggregate_patch_features(he_patch_features, method=self.patch_aggregation)
        mpm_roi_feature = self.mpm_extractor.aggregate_patch_features(mpm_patch_features, method=self.patch_aggregation)

        # See README.md for project setup and required files.
        he_roi_feature = he_roi_feature.to(self.device)
        mpm_roi_feature = mpm_roi_feature.to(self.device)

        # See README.md for project setup and required files.
        tacs_tensor = torch.tensor([tacs_type], dtype=torch.long, device=self.device)

        # See README.md for project setup and required files.
        if self.use_tacs_aware:
            fused_feature = self.fusion_module(
                he_roi_feature.unsqueeze(0),
                mpm_roi_feature.unsqueeze(0),
                tacs_tensor
            ).squeeze(0)
        else:
            fused_feature = self.fusion_module(
                he_roi_feature.unsqueeze(0),
                mpm_roi_feature.unsqueeze(0)
            ).squeeze(0)

        return {
            'he_patch_features': he_patch_features,
            'mpm_patch_features': mpm_patch_features,
            'he_roi_feature': he_roi_feature,
            'mpm_roi_feature': mpm_roi_feature,
            'fused_feature': fused_feature,
            'he_positions': he_positions,
            'mpm_positions': mpm_positions,
            'num_he_patches': len(he_patches),
            'num_mpm_patches': len(mpm_patches)
        }

    def forward(self,
                he_images: List[Image.Image],
                mpm_images: List[Image.Image],
                tacs_types: torch.Tensor) -> Dict[str, torch.Tensor]:
        """See README.md for the script purpose and required inputs."""

        batch_size = len(he_images)

        he_roi_features = []
        mpm_roi_features = []
        fused_features = []
        he_patch_features_list = []
        mpm_patch_features_list = []

        for i in range(batch_size):
            result = self.process_roi(
                he_images[i],
                mpm_images[i],
                tacs_types[i].item()
            )

            he_roi_features.append(result['he_roi_feature'])
            mpm_roi_features.append(result['mpm_roi_feature'])
            fused_features.append(result['fused_feature'])
            he_patch_features_list.append(result['he_patch_features'])
            mpm_patch_features_list.append(result['mpm_patch_features'])

        # See README.md for project setup and required files.
        he_roi_features = torch.stack(he_roi_features)
        mpm_roi_features = torch.stack(mpm_roi_features)
        fused_features = torch.stack(fused_features)

        return {
            'he_patch_features': he_patch_features_list,
            'mpm_patch_features': mpm_patch_features_list,
            'he_roi_features': he_roi_features,
            'mpm_roi_features': mpm_roi_features,
            'fused_features': fused_features
        }


class BatchFeatureExtractor:
    """See README.md for the script purpose and required inputs."""

    def __init__(self,
                 model: MultimodalFusionModel,
                 device: torch.device = None,
                 patch_batch_size: int = 256):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = model.to(self.device)
        self.model.eval()
        self.patch_batch_size = patch_batch_size

    def load_image(self, image_path: str) -> Image.Image:
        """See README.md for the script purpose and required inputs."""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Status {image_path}: {e}")
            # See README.md for project setup and required files.
            return Image.new('RGB', (512, 512), color='white')

    def extract_from_metadata(self,
                              metadata_path: str,
                              batch_size: int = 64,
                              output_dir: str = './xxx/xx',
                              save_patch_features: bool = False) -> Dict[str, List[ROIEmbedding]]:
        """See README.md for the script purpose and required inputs."""

        # See README.md for project setup and required files.
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"Status: Status {len(metadata.get('splits', {}))} Status")

        # See README.md for project setup and required files.
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # See README.md for project setup and required files.
        split_dirs = {}
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split_name] = split_dir
            print(f"Status: {split_dir}")

        all_embeddings = {'train': [], 'val': [], 'test': []}

        # See README.md for project setup and required files.
        for split_name, split_data in metadata.get('splits', {}).items():
            if split_name not in ['train', 'val', 'test']:
                print(f"Status: {split_name}")
                continue

            print(f"\nStatus {split_name} Status: {len(split_data.get('patients', []))} Status")

            split_output_dir = split_dirs[split_name]
            split_embeddings = []

            # See README.md for project setup and required files.
            for patient_data in tqdm(split_data['patients'], desc=f"Status{split_name}Status"):
                patient_id = patient_data['patient_id']
                patient_rois = patient_data['rois']

                patient_embeddings = []

                # See README.md for project setup and required files.
                for i in range(0, len(patient_rois), batch_size):
                    batch_rois = patient_rois[i:i + batch_size]

                    he_images = []
                    mpm_images = []
                    tacs_types = []
                    roi_info = []  # See README.md for project setup and required files.

                    for roi in batch_rois:
                        # See README.md for project setup and required files.
                        he_img = self.load_image(roi['he_path'])
                        he_images.append(he_img)

                        # See README.md for project setup and required files.
                        mpm_img = self.load_image(roi['mpm_path'])
                        mpm_images.append(mpm_img)

                        # See README.md for project setup and required files.
                        tacs_types.append(roi['tacs_type'])

                        # See README.md for project setup and required files.
                        roi_info.append({
                            'patient_id': patient_id,
                            'filename': roi['filename'],
                            'he_path': roi['he_path'],
                            'mpm_path': roi['mpm_path']
                        })

                    if he_images:
                        # See README.md for project setup and required files.
                        tacs_batch = torch.tensor(tacs_types, dtype=torch.long).to(self.device)

                        # See README.md for project setup and required files.
                        with torch.no_grad():
                            outputs = self.model(he_images, mpm_images, tacs_batch)

                        # See README.md for project setup and required files.
                        for j in range(len(batch_rois)):
                            embedding = ROIEmbedding(
                                patient_id=patient_id,
                                tacs_type=tacs_types[j],
                                he_feature=outputs['he_roi_features'][j].cpu().numpy(),
                                mpm_feature=outputs['mpm_roi_features'][j].cpu().numpy(),
                                fused_feature=outputs['fused_features'][j].cpu().numpy(),
                                he_path=roi_info[j]['he_path'],
                                mpm_path=roi_info[j]['mpm_path'],
                                filename=roi_info[j]['filename'],
                                split=split_name,
                                he_patch_features=outputs['he_patch_features'][
                                    j].cpu().numpy() if save_patch_features else None,
                                mpm_patch_features=outputs['mpm_patch_features'][
                                    j].cpu().numpy() if save_patch_features else None
                            )
                            patient_embeddings.append(embedding)

                split_embeddings.extend(patient_embeddings)

                # See README.md for project setup and required files.
                self.save_patient_features(patient_id, patient_embeddings, split_output_dir)

            all_embeddings[split_name] = split_embeddings

            # See README.md for project setup and required files.
            self.save_split_features(split_name, split_embeddings, split_output_dir)

        # See README.md for project setup and required files.
        self.save_all_features(all_embeddings, output_dir)

        # See README.md for project setup and required files.
        self.print_statistics(all_embeddings)

        return all_embeddings

    def save_patient_features(self,
                              patient_id: str,
                              embeddings: List[ROIEmbedding],
                              output_dir: Path):
        """See README.md for the script purpose and required inputs."""

        if not embeddings:
            return

        # See README.md for project setup and required files.
        fused_features = np.array([emb.fused_feature for emb in embeddings])
        he_features = np.array([emb.he_feature for emb in embeddings])
        mpm_features = np.array([emb.mpm_feature for emb in embeddings])
        tacs_types = np.array([emb.tacs_type for emb in embeddings])
        filenames = [emb.filename for emb in embeddings]

        # See README.md for project setup and required files.
        patient_file = output_dir / f"{patient_id}_features.npz"
        np.savez(patient_file,
                 fused_features=fused_features,
                 he_features=he_features,
                 mpm_features=mpm_features,
                 tacs_types=tacs_types,
                 filenames=filenames)

    def save_split_features(self,
                            split_name: str,
                            embeddings: List[ROIEmbedding],
                            output_dir: Path):
        """See README.md for the script purpose and required inputs."""

        if not embeddings:
            return

        # See README.md for project setup and required files.
        fused_features = np.array([emb.fused_feature for emb in embeddings])
        he_features = np.array([emb.he_feature for emb in embeddings])
        mpm_features = np.array([emb.mpm_feature for emb in embeddings])
        tacs_types = np.array([emb.tacs_type for emb in embeddings])
        filenames = [emb.filename for emb in embeddings]
        patient_ids = [emb.patient_id for emb in embeddings]

        # See README.md for project setup and required files.
        split_file = output_dir / f"{split_name}_all_features.npz"
        np.savez(split_file,
                 fused_features=fused_features,
                 he_features=he_features,
                 mpm_features=mpm_features,
                 tacs_types=tacs_types,
                 filenames=filenames,
                 patient_ids=patient_ids)

        print(f"✓ {split_name}Status: {len(embeddings)} StatusROI")

    def save_all_features(self,
                          all_embeddings: Dict[str, List[ROIEmbedding]],
                          output_dir: Path):
        """See README.md for the script purpose and required inputs."""

        # See README.md for project setup and required files.
        all_embeddings_list = []
        for split_name in ['train', 'val', 'test']:
            all_embeddings_list.extend(all_embeddings[split_name])

        if not all_embeddings_list:
            return

        # See README.md for project setup and required files.
        features_data = []
        for emb in all_embeddings_list:
            features_data.append({
                'patient_id': emb.patient_id,
                'tacs_type': emb.tacs_type,
                'filename': emb.filename,
                'he_path': emb.he_path,
                'mpm_path': emb.mpm_path,
                'split': emb.split,
                'he_feature': emb.he_feature.tolist(),
                'mpm_feature': emb.mpm_feature.tolist(),
                'fused_feature': emb.fused_feature.tolist()
            })

        # See README.md for project setup and required files.
        json_file = output_dir / 'all_features_summary.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(features_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Status: {json_file}")

    def print_statistics(self, all_embeddings: Dict[str, List[ROIEmbedding]]):
        """See README.md for the script purpose and required inputs."""
        print("\n" + "=" * 60)
        print("Processing...")
        print("=" * 60)

        total_rois = 0
        for split_name in ['train', 'val', 'test']:
            split_count = len(all_embeddings[split_name])
            total_rois += split_count
            print(f"{split_name}Status: {split_count} StatusROI")

        print(f"Status: {total_rois} StatusROI")

        # See README.md for project setup and required files.
        if total_rois > 0:
            first_emb = all_embeddings['train'][0] if all_embeddings['train'] else (
                all_embeddings['val'][0] if all_embeddings['val'] else (
                    all_embeddings['test'][0] if all_embeddings['test'] else None
                )
            )

            if first_emb:
                print(f"\nStatus:")
                print(f"  HEStatus: {len(first_emb.he_feature)} (Statusget_encoder.pyStatusUNIStatus1024Status)")
                print(f"  MPMStatus: {len(first_emb.mpm_feature)} (EfficientNet-B3Status1024Status)")
                print(f"  Status: {len(first_emb.fused_feature)} (Status256Status)")

        print("=" * 60)


def extract_features_from_processed_data(
        config_path: str = "./xxx/xx"):
    """See README.md for the script purpose and required inputs."""

    import yaml

    # See README.md for project setup and required files.
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    feature_config = config['feature_extraction']
    data_config = config['data']

    # See README.md for project setup and required files.
    device = torch.device(feature_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Status: {device}")

    # See README.md for project setup and required files.
    print("Processing...")
    model = MultimodalFusionModel(
        he_model_type=feature_config.get('he_model', 'uni'),
        mpm_model_type=feature_config.get('mpm_model', 'efficientnet_b3'),
        fusion_output_dim=feature_config.get('fusion_dim', 256),
        fusion_method=feature_config.get('fusion_method', 'cross_attention'),
        use_tacs_aware=feature_config.get('use_tacs_aware', True),
        patch_size=feature_config.get('patch_size', 224),
        patch_stride=feature_config.get('patch_stride', 224),
        patch_aggregation=feature_config.get('patch_aggregation', 'mean'),
        device=device
    )

    # See README.md for project setup and required files.
    extractor = BatchFeatureExtractor(
        model=model,
        device=device,
        patch_batch_size=feature_config.get('patch_batch_size', 256)
    )

    # See README.md for project setup and required files.
    metadata_path = Path(data_config.get('processed_data_dir', '256_processed_data')) / 'dataset_metadata.json'

    if not metadata_path.exists():
        # See README.md for project setup and required files.
        metadata_path = Path(data_config.get('processed_data_dir', '256_processed_data')) / 'roi_metadata.json'
        if not metadata_path.exists():
            print(f"Status: Status: {metadata_path}")
            return None

    print(f"Status: {metadata_path}")

    all_embeddings = extractor.extract_from_metadata(
        metadata_path=str(metadata_path),
        batch_size=feature_config.get('batch_size', 64),
        output_dir=feature_config.get('output_dir', './xxx/xx'),
        save_patch_features=feature_config.get('save_patch_features', False)
    )

    return all_embeddings


def main():
    """See README.md for the script purpose and required inputs."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Processing...')
    parser.add_argument('--config', type=str, default='./config-UNI-feature_crossattn.yaml',
                        help='Processing...')
    parser.add_argument('--metadata', type=str,
                        help='Processing...')
    # Example output directory for per-ROI fused feature NPZ archives.
    parser.add_argument('--output_dir', type=str, default='./xxx/xx/fused_features',
                        help='Processing...')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Processing...')
    parser.add_argument('--patch_size', type=int, default=224,
                        help='Processing...')
    parser.add_argument('--patch_stride', type=int, default=224,
                        help='Processing...')
    parser.add_argument('--fusion_method', type=str,
                        choices=['cross_attention', 'concat', 'add', 'attention', 'weighted_sum'],
                        default='cross_attention',
                        help='Processing...')
    parser.add_argument('--use_tacs_aware', action='store_true',
                        help='Processing...')
    parser.add_argument('--fusion_dim', type=int, default=256,
                        help='Processing...')
    parser.add_argument('--save_patch_features', action='store_true',
                        help='Processing...')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Processing...')
    parser.add_argument('--he_model', type=str, default='uni',
                        help='Processing...')

    args = parser.parse_args()

    print("=" * 60)
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("=" * 60)

    # See README.md for project setup and required files.
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Status: {config_path}")
        return

    # See README.md for project setup and required files.
    if args.metadata:
        metadata_path = args.metadata
    else:
        data_config = config.get('data', {})
        metadata_path = Path(data_config.get('processed_data_dir', '256_processed_data')) / 'dataset_metadata.json'

    if args.output_dir:
        config['feature_extraction']['output_dir'] = args.output_dir

    if args.batch_size:
        config['feature_extraction']['batch_size'] = args.batch_size

    if args.patch_size:
        config['feature_extraction']['patch_size'] = args.patch_size

    if args.patch_stride:
        config['feature_extraction']['patch_stride'] = args.patch_stride

    if args.fusion_method:
        config['feature_extraction']['fusion_method'] = args.fusion_method

    if args.use_tacs_aware:
        config['feature_extraction']['use_tacs_aware'] = True

    if args.fusion_dim:
        config['feature_extraction']['fusion_dim'] = args.fusion_dim

    if args.save_patch_features:
        config['feature_extraction']['save_patch_features'] = True

    if args.device:
        config['feature_extraction']['device'] = args.device

    if args.he_model:
        config['feature_extraction']['he_model'] = args.he_model

    print(f"Status: {config_path}")
    print(f"Status: {metadata_path}")
    print(f"Status: {config['feature_extraction']['output_dir']}")
    print(f"HEStatus: {config['feature_extraction'].get('he_model', 'uni')}")
    print(f"MPMStatus: {config['feature_extraction'].get('mpm_model', 'efficientnet_b3')}")
    print(f"ROIStatus: {config['feature_extraction']['batch_size']}")
    print(f"PatchStatus: {config['feature_extraction'].get('patch_size', 224)} (Status)")
    print(f"PatchStatus: {config['feature_extraction'].get('patch_stride', 224)} (Status)")
    print(f"Status: {config['feature_extraction'].get('fusion_method', 'cross_attention')}")
    print(f"TACSStatus: {config['feature_extraction'].get('use_tacs_aware', False)}")
    print(f"Status: {config['feature_extraction'].get('fusion_dim', 256)} (Status256Status)")
    print(f"StatusPatchStatus: {config['feature_extraction'].get('save_patch_features', False)}")
    print(f"Status: {config['feature_extraction'].get('device', 'cuda')}")
    print("=" * 60)

    # See README.md for project setup and required files.
    all_embeddings = extract_features_from_processed_data(config_path)

    if all_embeddings:
        print(f"\nStatus!")
        total_rois = sum(len(embeddings) for embeddings in all_embeddings.values())
        print(f"Status {total_rois} StatusROIStatus")
        print(f"Status: 256 (Status)")

        # See README.md for project setup and required files.
        print(f"\nStatus:")
        output_dir = Path(config['feature_extraction']['output_dir'])
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            if split_dir.exists():
                npz_files = list(split_dir.glob("*.npz"))
                print(f"  {split_dir}/: {len(npz_files)} Status")

        print(f"\nStatustrain/val/testStatus: {output_dir}")


if __name__ == "__main__":
    main()
