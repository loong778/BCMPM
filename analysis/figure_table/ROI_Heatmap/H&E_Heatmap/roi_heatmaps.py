"""See README.md for the script purpose and required inputs."""

import os
import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange
from scipy.stats import rankdata

# See README.md for project setup and required files.
try:
    import tifffile

    # Required input: see README.md for the required file and directory layout.

    HAS_TIFFFILE = True
except ImportError:
    # Required input: see README.md for the required file and directory layout.
    HAS_TIFFFILE = False

# See README.md for project setup and required files.
from hipt_4k import HIPT_4K
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms


# See README.md for project setup and required files.
def robust_read_image(image_path: str, max_size: int = 4096) -> Image.Image:
    """See README.md for the script purpose and required inputs."""
    ext = os.path.splitext(image_path)[1].lower()

    # See README.md for project setup and required files.
    if ext in ('.tif', '.tiff') and HAS_TIFFFILE:
        img_array = tifffile.imread(image_path)

        # See README.md for project setup and required files.
        if img_array.ndim == 3 and img_array.shape[0] < 5:  # See README.md for project setup and required files.
            if img_array.shape[0] <= 4:
                img_array = np.transpose(img_array, (1, 2, 0))
        elif img_array.ndim == 4:  # See README.md for project setup and required files.
            img_array = img_array[0]

        # See README.md for project setup and required files.
        if img_array.dtype == np.uint16:
            img_array = (img_array / 256).astype(np.uint8)
        elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            # See README.md for project setup and required files.
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)

        # See README.md for project setup and required files.
        if img_array.ndim == 2:  # See README.md for project setup and required files.
            img = Image.fromarray(img_array, mode='L').convert('RGB')
        elif img_array.shape[2] == 4:  # RGBA
            img = Image.fromarray(img_array, mode='RGBA').convert('RGB')
        else:  # See README.md for project setup and required files.
            img = Image.fromarray(img_array[:, :, :3], mode='RGB')
    else:
        # See README.md for project setup and required files.
        img = Image.open(image_path).convert('RGB')

    # See README.md for project setup and required files.
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Status: {w}x{h} -> {new_w}x{new_h}")

    return img


# See README.md for project setup and required files.
def concat_scores256(attns, w_256, h_256):
    """See README.md for the script purpose and required inputs."""
    rank = lambda v: rankdata(v) * 100 / len(v)
    color_block = [rank(attn.flatten()).reshape(16, 16) for attn in attns]
    rows = []
    for i in range(h_256):
        row = np.concatenate(color_block[i * w_256:(i + 1) * w_256], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)  # shape: (16*h_256, 16*w_256)


def concat_scores4k(attn, size=(16, 16)):
    """See README.md for the script purpose and required inputs."""
    rank = lambda v: rankdata(v) * 100 / len(v)
    return rank(attn.flatten()).reshape(size)


def get_attention_maps(image: Image.Image,
                       model256: torch.nn.Module,
                       model4k: torch.nn.Module,
                       device256='cuda',
                       device4k='cuda'):
    """See README.md for the script purpose and required inputs."""
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = t(image).unsqueeze(0)  # [1, C, H, W]

    # See README.md for project setup and required files.
    _, _, h, w = x.shape
    new_h = (h // 256) * 256
    new_w = (w // 256) * 256
    if new_h == 0 or new_w == 0:
        raise ValueError(f"Image size {w}x{h} is too small (min 256x256 required)")
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    x = x[:, :, top:top + new_h, left:left + new_w]

    # See README.md for project setup and required files.
    patches = x.unfold(2, 256, 256).unfold(3, 256, 256)
    w_256 = patches.shape[3]  # See README.md for project setup and required files.
    h_256 = patches.shape[2]  # See README.md for project setup and required files.
    patches = rearrange(patches, 'b c p1 p2 w h -> (b p1 p2) c w h')  # [B, 3, 256, 256]
    B = patches.shape[0]

    # See README.md for project setup and required files.
    features_cls256_list = []
    attn_256_list = []
    mini_batch_size = 256

    for start in range(0, B, mini_batch_size):
        end = min(start + mini_batch_size, B)
        mini_patches = patches[start:end].to(device256, non_blocking=True)

        with torch.no_grad():
            mini_feat = model256(mini_patches)
            mini_attn = model256.get_last_selfattention(mini_patches)
        mini_feat = mini_feat.detach().cpu()
        # [mini_batch, nh, 1+N, 1+N] -> [mini_batch, nh, N] (N=256)
        mini_attn = mini_attn[:, :, 0, 1:].detach().cpu()
        features_cls256_list.append(mini_feat)
        attn_256_list.append(mini_attn)

    features_cls256 = torch.cat(features_cls256_list, dim=0)  # [B, 384]
    attn_256_all = torch.cat(attn_256_list, dim=0)  # [B, nh, 256]

    nh = attn_256_all.shape[1]
    attn_256 = attn_256_all.reshape(B, nh, 16, 16).numpy()

    # See README.md for project setup and required files.
    features_4k_grid = rearrange(features_cls256, '(h w) d -> 1 d h w', h=h_256, w=w_256)
    features_4k_grid = features_4k_grid.to(device4k, non_blocking=True)

    with torch.no_grad():
        _ = model4k(features_4k_grid)
        attn_4k = model4k.get_last_selfattention(features_4k_grid)  # [1, nh, 1+N, 1+N]
    attn_4k = attn_4k[0, :, 0, 1:]  # [nh, N]
    attn_4k = attn_4k.reshape(nh, h_256, w_256).cpu().numpy()

    # See README.md for project setup and required files.
    attn_256_maps = []
    for head in range(nh):
        head_attn = attn_256[:, head, :, :]
        full_map = concat_scores256(head_attn, w_256, h_256)
        attn_256_maps.append(full_map)
    attn_256_maps = np.stack(attn_256_maps, axis=0)  # [nh, new_h, new_w]

    attn_4k_maps = []
    for head in range(nh):
        head_attn = attn_4k[head]
        full_map = concat_scores4k(head_attn, size=(h_256, w_256))
        full_map = cv2.resize(full_map, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        attn_4k_maps.append(full_map)
    attn_4k_maps = np.stack(attn_4k_maps, axis=0)  # [nh, new_h, new_w]

    return attn_256_maps, attn_4k_maps


def overlay_heatmap(image: Image.Image,
                    attn_map: np.ndarray,
                    colormap=plt.get_cmap('coolwarm'),
                    alpha=0.5) -> Image.Image:
    """See README.md for the script purpose and required inputs."""
    np_img = np.array(image.convert('RGB'))
    h, w = np_img.shape[:2]

    if attn_map.shape[0] != h or attn_map.shape[1] != w:
        attn_map_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        attn_map_resized = attn_map

    if attn_map_resized.max() > attn_map_resized.min():
        attn_norm = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min())
    else:
        attn_norm = np.zeros_like(attn_map_resized)

    colored = (colormap(attn_norm)[:, :, :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(np_img, 1 - alpha, colored, alpha, 0)
    return Image.fromarray(blended)


def process_single_image(image_path: str,
                         output_dir: str,
                         model256: torch.nn.Module,
                         model4k: torch.nn.Module,
                         device256='cuda:0',
                         device4k='cuda:0',
                         head_256=None,
                         head_4k=None,
                         alpha=0.5,
                         max_image_size=4096):
    """See README.md for the script purpose and required inputs."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # See README.md for project setup and required files.
    img = robust_read_image(image_path, max_size=max_image_size)
    print(f"  Status: {img.size}")

    # See README.md for project setup and required files.
    attn_256, attn_4k = get_attention_maps(img, model256, model4k, device256, device4k)

    if head_256 is not None:
        map_256 = attn_256[head_256]
    else:
        map_256 = np.mean(attn_256, axis=0)

    if head_4k is not None:
        map_4k = attn_4k[head_4k]
    else:
        map_4k = np.mean(attn_4k, axis=0)

    cmap = plt.get_cmap('coolwarm')
    blended_256 = overlay_heatmap(img, map_256, colormap=cmap, alpha=alpha)
    blended_4k = overlay_heatmap(img, map_4k, colormap=cmap, alpha=alpha)

    blended_256.save(os.path.join(output_dir, f'{base_name}_vit256_heatmap.png'))
    blended_4k.save(os.path.join(output_dir, f'{base_name}_vit4k_heatmap.png'))
    print(f"  Status: {output_dir}")
    return True


def batch_process_tif_folder(image_dir: str,
                             output_root: str,
                             model256: torch.nn.Module,
                             model4k: torch.nn.Module,
                             device256='cuda:0',
                             device4k='cuda:0',
                             head_256=None,
                             head_4k=None,
                             alpha=0.5,
                             max_image_size=4096):
    """See README.md for the script purpose and required inputs."""
    # See README.md for project setup and required files.
    tif_files = glob.glob(os.path.join(image_dir, '*.tif'))
    tif_files += glob.glob(os.path.join(image_dir, '*.tiff'))
    tif_files = sorted(list(set(tif_files)))  # See README.md for project setup and required files.

    if not tif_files:
        print(f"Status '{image_dir}' Status .tif Status")
        return

    print(f"Status {len(tif_files)} Status .tif Status...\n")

    for idx, tif_path in enumerate(tif_files, 1):
        basename = os.path.splitext(os.path.basename(tif_path))[0]
        print(f"[{idx}/{len(tif_files)}] Status: {basename}")

        # See README.md for project setup and required files.
        output_subdir = os.path.join(output_root, basename)
        try:
            process_single_image(
                image_path=tif_path,
                output_dir=output_subdir,
                model256=model256,
                model4k=model4k,
                device256=device256,
                device4k=device4k,
                head_256=head_256,
                head_4k=head_4k,
                alpha=alpha,
                max_image_size=max_image_size
            )
        except Exception as e:
            print(f"  Status: {e}\n")
        print("-" * 60)


# See README.md for project setup and required files.
if __name__ == '__main__':
    # See README.md for project setup and required files.
    # Required input: see README.md for the required file and directory layout.
    MODEL256_PATH = './xxx/xx'
    # Required input: see README.md for the required file and directory layout.
    MODEL4K_PATH = './xxx/xx'
    # Required input: see README.md for the required file and directory layout.
    IMAGE_DIR = './xxx/xx'  # See README.md for project setup and required files.
    # Example output path; generated files are written here.
    OUTPUT_ROOT = './xxx/xx'  # Example output directory for batch heatmaps.

    # See README.md for project setup and required files.
    print("Processing...")
    model256 = get_vit256(pretrained_weights=MODEL256_PATH).to('cuda:0')
    model4k = get_vit4k(pretrained_weights=MODEL4K_PATH).to('cuda:0')
    model256.eval()
    model4k.eval()
    print("Processing...")

    # See README.md for project setup and required files.
    batch_process_tif_folder(
        image_dir=IMAGE_DIR,
        output_root=OUTPUT_ROOT,
        model256=model256,
        model4k=model4k,
        device256='cuda:0',
        device4k='cuda:0',
        head_256=None,
        head_4k=None,
        alpha=0.5,
        max_image_size=9182  # See README.md for project setup and required files.
    )

    print("Processing...")
