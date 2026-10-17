"""See README.md for the script purpose and required inputs."""
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import glob
import re
from pathlib import Path
from dataclasses import dataclass
import warnings
import random

warnings.filterwarnings('ignore')
# See README.md for project setup and required files.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class ROIImage:
    """See README.md for the script purpose and required inputs."""
    patient_id: str  # See README.md for project setup and required files.
    image_type: str  # See README.md for project setup and required files.
    tacs_type: int  # See README.md for project setup and required files.
    image_path: str  # See README.md for project setup and required files.
    filename: str  # See README.md for project setup and required files.


class PairedTransform:
    """See README.md for the script purpose and required inputs."""

    def __init__(self, augment=False):
        self.augment = augment

        # See README.md for project setup and required files.
        self.base_transform = self._create_base_transform()
        if augment:
            self.augment_transform = self._create_augment_transform()
        else:
            self.augment_transform = self.base_transform

    def _create_base_transform(self):
        """See README.md for the script purpose and required inputs."""
        return T.Compose([
            T.ToTensor(),  # See README.md for project setup and required files.
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _create_augment_transform(self):
        """See README.md for the script purpose and required inputs."""
        return T.Compose([
            # See README.md for project setup and required files.
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.3, hue=0.1),
            # See README.md for project setup and required files.
            T.ToTensor(),
            # See README.md for project setup and required files.
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1), value='random')
        ])

    def __call__(self, he_image, mpm_image):
        """See README.md for the script purpose and required inputs."""
        if not isinstance(he_image, Image.Image) or not isinstance(mpm_image, Image.Image):
            raise TypeError(f"StatusPIL.ImageStatushe_imageStatus{type(he_image)}Statusmpm_imageStatus{type(mpm_image)}")

        if self.augment:
            # See README.md for project setup and required files.
            python_random_state = random.getstate()
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()

            try:
                # See README.md for project setup and required files.
                he_transformed = self.augment_transform(he_image)

                # See README.md for project setup and required files.
                random.setstate(python_random_state)
                np.random.set_state(numpy_random_state)
                torch.set_rng_state(torch_random_state)

                mpm_transformed = self.augment_transform(mpm_image)
            except Exception as e:
                print(f"Status: {e}")
                # See README.md for project setup and required files.
                he_transformed = self.base_transform(he_image)
                mpm_transformed = self.base_transform(mpm_image)
        else:
            # See README.md for project setup and required files.
            he_transformed = self.base_transform(he_image)
            mpm_transformed = self.base_transform(mpm_image)

        return he_transformed, mpm_transformed


class ROIDataProcessor:
    """See README.md for the script purpose and required inputs."""

    def __init__(self,
                 dataset_root: str,  # See README.md for project setup and required files.
                 tacs_pattern: str = r'C(\d+)'):
        """See README.md for the script purpose and required inputs."""
        self.dataset_root = Path(dataset_root)
        self.tacs_pattern = tacs_pattern

        # See README.md for project setup and required files.
        self.train_transform = PairedTransform(augment=True)
        self.val_transform = PairedTransform(augment=False)

        # See README.md for project setup and required files.
        self.image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

        print(f"✓ Status")
        print(f"✓ Status: {self.dataset_root}")
        print(f"✓ StatusTACSStatus")

    def parse_tacs_type_from_filename(self, filename: str) -> List[int]:
        """See README.md for the script purpose and required inputs."""
        # See README.md for project setup and required files.
        basename = Path(filename).stem

        tacs_types = []

        # See README.md for project setup and required files.
        if '+' in basename:
            # See README.md for project setup and required files.
            matches = re.findall(r'C(\d+)', basename, re.IGNORECASE)
            for match in matches:
                try:
                    tacs_type = int(match)
                    if 1 <= tacs_type <= 8 and tacs_type not in tacs_types:
                        tacs_types.append(tacs_type)
                except ValueError:
                    continue

        # See README.md for project setup and required files.
        if not tacs_types:
            matches = re.findall(self.tacs_pattern, basename, re.IGNORECASE)
            for match in matches:
                try:
                    tacs_type = int(match)
                    if 1 <= tacs_type <= 8 and tacs_type not in tacs_types:
                        tacs_types.append(tacs_type)
                except ValueError:
                    continue

        # See README.md for project setup and required files.
        if not tacs_types:
            # See README.md for project setup and required files.
            all_numbers = re.findall(r'\d+', basename)
            for num_str in all_numbers:
                try:
                    tacs_type = int(num_str)
                    if 1 <= tacs_type <= 8 and tacs_type not in tacs_types:
                        tacs_types.append(tacs_type)
                except ValueError:
                    continue

        # See README.md for project setup and required files.
        tacs_types = sorted(list(set(tacs_types)))

        if not tacs_types:
            print(f"Status: Status {filename} StatusTACSStatus")

        return tacs_types

    def load_split_data(self, split_name: str) -> Dict[str, List[ROIImage]]:
        """See README.md for the script purpose and required inputs."""
        split_dir = self.dataset_root / split_name

        if not split_dir.exists():
            print(f"Status: Status: {split_dir}")
            return {}

        he_dir = split_dir / 'he'
        mpm_dir = split_dir / 'mpm'

        if not he_dir.exists() or not mpm_dir.exists():
            print(f"Status: HEStatusMPMStatus: {he_dir}, {mpm_dir}")
            return {}

        print(f"\nStatus {split_name} Status...")

        # See README.md for project setup and required files.
        he_patient_folders = [d for d in he_dir.iterdir() if d.is_dir()]
        mpm_patient_folders = [d for d in mpm_dir.iterdir() if d.is_dir()]

        # See README.md for project setup and required files.
        he_patient_ids = {f.name: f for f in he_patient_folders}
        mpm_patient_ids = {f.name: f for f in mpm_patient_folders}
        common_patient_ids = set(he_patient_ids.keys()) & set(mpm_patient_ids.keys())

        print(f"  HEStatus: {len(he_patient_ids)}, MPMStatus: {len(mpm_patient_ids)}")
        print(f"  Status: {len(common_patient_ids)}")

        if len(common_patient_ids) == 0:
            print(f"  Status: {split_name} StatusHE/MPMStatus")
            return {}

        # See README.md for project setup and required files.
        split_data = {}

        for patient_id in common_patient_ids:
            he_patient_path = he_patient_ids[patient_id]
            mpm_patient_path = mpm_patient_ids[patient_id]

            # See README.md for project setup and required files.
            he_image_files = []
            for ext in self.image_extensions:
                he_image_files.extend(list(he_patient_path.glob(f"*{ext}")))

            mpm_image_files = []
            for ext in self.image_extensions:
                mpm_image_files.extend(list(mpm_patient_path.glob(f"*{ext}")))

            # See README.md for project setup and required files.
            he_files_by_name = {f.name: f for f in he_image_files}
            mpm_files_by_name = {f.name: f for f in mpm_image_files}
            common_filenames = set(he_files_by_name.keys()) & set(mpm_files_by_name.keys())

            if not common_filenames:
                # See README.md for project setup and required files.
                continue

            # See README.md for project setup and required files.
            patient_rois = []
            for filename in common_filenames:
                tacs_types = self.parse_tacs_type_from_filename(filename)
                if not tacs_types:
                    print(f"  Status: StatusTACSStatus: {filename}")
                    continue

                # See README.md for project setup and required files.
                for tacs_type in tacs_types:
                    # HE ROI
                    he_roi = ROIImage(
                        patient_id=patient_id,
                        image_type='HE',
                        tacs_type=tacs_type,
                        image_path=str(he_files_by_name[filename]),
                        filename=filename
                    )

                    # MPM ROI
                    mpm_roi = ROIImage(
                        patient_id=patient_id,
                        image_type='MPM',
                        tacs_type=tacs_type,
                        image_path=str(mpm_files_by_name[filename]),
                        filename=filename
                    )

                    # See README.md for project setup and required files.
                    patient_rois.extend([he_roi, mpm_roi])

                    if len(tacs_types) > 1:
                        print(f"    Status: {filename} -> TACS={tacs_type}")

            if patient_rois:
                split_data[patient_id] = patient_rois
                total_pairs = len(patient_rois) // 2
                print(f"  Status {patient_id}: {total_pairs} StatusROIStatus")

        total_roi_pairs = sum(len(rois) for rois in split_data.values()) // 2
        print(f"  {split_name} Status: {len(split_data)} Status, {total_roi_pairs} StatusROIStatus")

        return split_data

    def create_all_datasets(self) -> Tuple[Dict, Dict, Dict]:
        """See README.md for the script purpose and required inputs."""
        print("=" * 60)
        print("Processing...")
        print("=" * 60)

        # See README.md for project setup and required files.
        train_data = self.load_split_data('train')
        val_data = self.load_split_data('val')
        test_data = self.load_split_data('test')

        # See README.md for project setup and required files.
        total_patients = len(train_data) + len(val_data) + len(test_data)
        total_roi_pairs = sum(len(rois) for rois in train_data.values()) // 2
        total_roi_pairs += sum(len(rois) for rois in val_data.values()) // 2
        total_roi_pairs += sum(len(rois) for rois in test_data.values()) // 2

        print(f"\nStatus:")
        print(f"  Status: {len(train_data)} Status, "
              f"{sum(len(rois) for rois in train_data.values()) // 2} StatusROIStatus")
        print(f"  Status: {len(val_data)} Status, "
              f"{sum(len(rois) for rois in val_data.values()) // 2} StatusROIStatus")
        print(f"  Status: {len(test_data)} Status, "
              f"{sum(len(rois) for rois in test_data.values()) // 2} StatusROIStatus")
        print(f"  Status: {total_patients} Status, {total_roi_pairs} StatusROIStatus")
        print("=" * 60)

        return train_data, val_data, test_data

    def save_metadata_multi_label(self,
                                  train_data: Dict[str, List[ROIImage]],
                                  val_data: Dict[str, List[ROIImage]],
                                  test_data: Dict[str, List[ROIImage]],
                                  output_dir: str):
        """See README.md for the script purpose and required inputs."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        metadata = {
            'splits': {},
            'description': 'Processing...',
            'version': '1.2'
        }

        for split_name, split_dict in all_data.items():
            split_metadata = {
                'patients': []
            }

            tacs_counts = {}

            for patient_id, rois in split_dict.items():
                patient_rois = []

                # See README.md for project setup and required files.
                file_groups = {}
                for roi in rois:
                    if roi.filename not in file_groups:
                        file_groups[roi.filename] = {
                            'he_path': None,
                            'mpm_path': None,
                            'tacs_types': set()
                        }

                    if roi.image_type == 'HE':
                        file_groups[roi.filename]['he_path'] = roi.image_path
                    else:  # MPM
                        file_groups[roi.filename]['mpm_path'] = roi.image_path

                    file_groups[roi.filename]['tacs_types'].add(roi.tacs_type)

                # See README.md for project setup and required files.
                for filename, file_info in file_groups.items():
                    if file_info['he_path'] and file_info['mpm_path']:
                        tacs_types_list = sorted(list(file_info['tacs_types']))

                        patient_rois.append({
                            'filename': filename,
                            'tacs_type': tacs_types_list,  # See README.md for project setup and required files.
                            'he_path': file_info['he_path'],
                            'mpm_path': file_info['mpm_path']
                        })

                        # See README.md for project setup and required files.
                        for tacs_type in tacs_types_list:
                            tacs_counts[tacs_type] = tacs_counts.get(tacs_type, 0) + 1

                if patient_rois:
                    split_metadata['patients'].append({
                        'patient_id': patient_id,
                        'num_rois': len(patient_rois),
                        'rois': patient_rois
                    })

            metadata['splits'][split_name] = split_metadata
            metadata['splits'][split_name]['tacs_distribution'] = tacs_counts

        # See README.md for project setup and required files.
        metadata_file = output_path / 'dataset_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✓ Status: {metadata_file}")
        print(f"✓ Statustacs_typeStatus")

        # See README.md for project setup and required files.
        csv_data = []
        for split_name, split_metadata in metadata['splits'].items():
            for patient in split_metadata.get('patients', []):
                for roi in patient.get('rois', []):
                    csv_data.append({
                        'split': split_name,
                        'patient_id': patient['patient_id'],
                        'filename': roi['filename'],
                        'tacs_type': ','.join(map(str, roi['tacs_type'])) if isinstance(roi['tacs_type'],
                                                                                        list) else str(
                            roi['tacs_type']),
                        'tacs_type_array': roi['tacs_type'],
                        'he_path': roi['he_path'],
                        'mpm_path': roi['mpm_path']
                    })

        df = pd.DataFrame(csv_data)
        csv_file = output_path / 'dataset_metadata.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✓ CSVStatus: {csv_file}")

        # See README.md for project setup and required files.
        self.save_dataset_statistics(metadata, output_path)

        return metadata

    def save_dataset_statistics(self, metadata: Dict, output_dir: Path):
        """See README.md for the script purpose and required inputs."""
        stats_file = output_dir / 'dataset_statistics.txt'

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("Processing...")
            f.write("=" * 60 + "\n\n")

            f.write("Processing...")
            f.write("-" * 30 + "\n")

            total_rois_all = 0
            for split_name in ['train', 'val', 'test']:
                if split_name in metadata['splits']:
                    split_info = metadata['splits'][split_name]
                    num_patients = len(split_info.get('patients', []))
                    num_rois = sum(p.get('num_rois', 0) for p in split_info.get('patients', []))
                    total_rois_all += num_rois

                    f.write(f"{split_name.capitalize()}Status:\n")
                    f.write(f"  Status: {num_patients}\n")
                    f.write(f"  ROIStatus: {num_rois}\n")

                    # See README.md for project setup and required files.
                    tacs_counts = split_info.get('tacs_distribution', {})
                    f.write(f"  TACSStatus:\n")
                    for tacs_type in sorted(tacs_counts.keys()):
                        count = tacs_counts[tacs_type]
                        percentage = (count / num_rois * 100) if num_rois > 0 else 0
                        f.write(f"    TACS{tacs_type}: {count} ({percentage:.1f}%)\n")

                    # See README.md for project setup and required files.
                    multi_label_count = 0
                    for patient in split_info.get('patients', []):
                        for roi in patient.get('rois', []):
                            if isinstance(roi.get('tacs_type', []), list) and len(roi['tacs_type']) > 1:
                                multi_label_count += 1

                    multi_label_percentage = (multi_label_count / num_rois * 100) if num_rois > 0 else 0
                    f.write(f"  Status: {multi_label_count} ({multi_label_percentage:.1f}%)\n\n")

            f.write(f"StatusROIStatus: {total_rois_all}\n")

        print(f"✓ Status: {stats_file}")


class PairedROIDataset(Dataset):
    """See README.md for the script purpose and required inputs."""

    def __init__(self,
                 split_data: Dict[str, List[ROIImage]],
                 transform=None,
                 phase: str = 'train'):
        """See README.md for the script purpose and required inputs."""
        self.phase = phase

        # See README.md for project setup and required files.
        if transform is None:
            if phase == 'train':
                self.transform = PairedTransform(augment=True)
            else:
                self.transform = PairedTransform(augment=False)
        else:
            self.transform = transform

        # See README.md for project setup and required files.
        self.samples = []

        for patient_id, rois in split_data.items():
            # See README.md for project setup and required files.
            he_rois = [roi for roi in rois if roi.image_type == 'HE']
            mpm_rois = [roi for roi in rois if roi.image_type == 'MPM']

            # See README.md for project setup and required files.
            for he_roi in he_rois:
                mpm_roi = next(
                    (r for r in mpm_rois if r.filename == he_roi.filename and r.tacs_type == he_roi.tacs_type), None)

                if mpm_roi:
                    self.samples.append({
                        'patient_id': patient_id,
                        'filename': he_roi.filename,
                        'tacs_type': he_roi.tacs_type,
                        'he_path': he_roi.image_path,
                        'mpm_path': mpm_roi.image_path,
                        'split': phase
                    })

        print(f"Status {phase} Status: {len(self.samples)} Status")
        print(f"Status: Statusresize")

    def __len__(self):
        """See README.md for the script purpose and required inputs."""
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            # See README.md for project setup and required files.
            he_image = Image.open(sample['he_path']).convert('RGB')

            # See README.md for project setup and required files.
            mpm_image = Image.open(sample['mpm_path']).convert('RGB')

            # See README.md for project setup and required files.
            he_size = he_image.size  # (width, height)
            mpm_size = mpm_image.size

            # See README.md for project setup and required files.
            if self.transform:
                he_transformed, mpm_transformed = self.transform(he_image, mpm_image)
            else:
                # See README.md for project setup and required files.
                he_transformed = TF.to_tensor(he_image)
                mpm_transformed = TF.to_tensor(mpm_image)

                # See README.md for project setup and required files.
                he_transformed = TF.normalize(he_transformed,
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                mpm_transformed = TF.normalize(mpm_transformed,
                                               mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

            return {
                'patient_id': sample['patient_id'],
                'filename': sample['filename'],
                'tacs_type': sample['tacs_type'],
                'he_image': he_transformed,
                'mpm_image': mpm_transformed,
                'he_path': sample['he_path'],
                'mpm_path': sample['mpm_path'],
                'split': sample['split'],
                'he_size': he_size,  # See README.md for project setup and required files.
                'mpm_size': mpm_size
            }

        except Exception as e:
            print(f"Status: {sample['he_path']}, Status: {e}")
            # See README.md for project setup and required files.
            # See README.md for project setup and required files.
            return {
                'patient_id': sample['patient_id'],
                'filename': sample['filename'],
                'tacs_type': sample['tacs_type'],
                'he_image': None,
                'mpm_image': None,
                'he_path': sample['he_path'],
                'mpm_path': sample['mpm_path'],
                'split': sample['split'],
                'he_size': (0, 0),
                'mpm_size': (0, 0)
            }


def create_data_loaders(config: Dict, verbose: bool = True):
    """See README.md for the script purpose and required inputs."""

    # See README.md for project setup and required files.
    processor = ROIDataProcessor(
        dataset_root=config['data']['dataset_root']
    )

    # See README.md for project setup and required files.
    if verbose:
        print("Processing...")
        print("=" * 60)
        print("Processing...")
        print("Processing...")

    train_data, val_data, test_data = processor.create_all_datasets()

    # See README.md for project setup and required files.
    if verbose:
        print("Processing...")

    train_dataset = PairedROIDataset(
        train_data,
        phase='train'
    )

    val_dataset = PairedROIDataset(
        val_data,
        phase='val'
    )

    test_dataset = PairedROIDataset(
        test_data,
        phase='test'
    )

    # See README.md for project setup and required files.
    def custom_collate_fn(batch):
        """See README.md for the script purpose and required inputs."""
        collated_batch = {}

        # See README.md for project setup and required files.
        for key in batch[0].keys():
            values = [item[key] for item in batch]

            # See README.md for project setup and required files.
            if key in ['he_image', 'mpm_image'] and values[0] is not None:
                # See README.md for project setup and required files.
                shapes = [v.shape for v in values if v is not None]
                if len(set(shapes)) == 1:
                    # See README.md for project setup and required files.
                    collated_batch[key] = torch.stack(values)
                else:
                    # See README.md for project setup and required files.
                    collated_batch[key] = values
            else:
                # See README.md for project setup and required files.
                collated_batch[key] = values

        return collated_batch

    # See README.md for project setup and required files.
    batch_size = config['data'].get('batch_size', 8)
    num_workers = config['data'].get('num_workers', 0)

    # See README.md for project setup and required files.
    import sys
    if sys.platform == 'win32':
        num_workers = 0
        if verbose:
            print("Processing...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # See README.md for project setup and required files.
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    if verbose:
        print(f"\nStatus:")
        print(f"  Status: {len(train_dataset)} Status")
        print(f"  Status: {len(val_dataset)} Status")
        print(f"  Status: {len(test_dataset)} Status")
        print(f"  Status: {batch_size}")
        print(f"  Status: Statusresize")
        print(f"  Status: StatuscollateStatus")

    # See README.md for project setup and required files.
    output_dir = config['data'].get('output_dir', './raw_processed_data')
    metadata = processor.save_metadata_multi_label(train_data, val_data, test_data, output_dir)

    return train_loader, val_loader, test_loader, metadata


def test_data_loading(train_loader, val_loader, test_loader):
    """See README.md for the script purpose and required inputs."""
    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)

    # See README.md for project setup and required files.
    print("Processing...")
    try:
        train_batch = next(iter(train_loader))
        print(f"  Status: {len(train_batch['he_image'])}")

        # See README.md for project setup and required files.
        he_images = train_batch['he_image']
        if isinstance(he_images, torch.Tensor):
            print(f"  HEStatus: {he_images.shape} (Status)")
        else:
            # See README.md for project setup and required files.
            print(f"  HEStatus: {len(he_images)}")
            for i, img in enumerate(he_images[:3]):
                if img is not None:
                    print(f"    HEStatus{i + 1}Status: {img.shape}")

        mpm_images = train_batch['mpm_image']
        if isinstance(mpm_images, torch.Tensor):
            print(f"  MPMStatus: {mpm_images.shape} (Status)")
        else:
            # See README.md for project setup and required files.
            print(f"  MPMStatus: {len(mpm_images)}")
            for i, img in enumerate(mpm_images[:3]):
                if img is not None:
                    print(f"    MPMStatus{i + 1}Status: {img.shape}")

        print(f"  TACSStatus: {train_batch['tacs_type'][:5]}")
        print(f"  StatusID: {train_batch['patient_id'][:3]}")
        print(f"  Status: {train_batch['split'][:3]}")

    except Exception as e:
        print(f"  Status: {e}")

    # See README.md for project setup and required files.
    print("Processing...")
    try:
        val_batch = next(iter(val_loader))
        print(f"  Status: {len(val_batch['he_image'])}")

    except Exception as e:
        print(f"  Status: {e}")

    # See README.md for project setup and required files.
    print("Processing...")
    try:
        test_batch = next(iter(test_loader))
        print(f"  Status: {len(test_batch['he_image'])}")

    except Exception as e:
        print(f"  Status: {e}")

    print("=" * 60)


def main():
    """See README.md for the script purpose and required inputs."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='Prepare paired H&E and MPM ROI data.')
    parser.add_argument('--config', type=str, default='./xxx/xx/preprocess.yaml', help='YAML config containing dataset_root and output_dir.')
    # Required input: train/val/test/he and train/val/test/mpm patient folders containing paired ROI images.
    parser.add_argument('--dataset_root', type=str, default='./xxx/xx/paired_rois', help='Root of the paired H&E and MPM ROI directory tree.')
    # Example output directory for metadata JSON/CSV and preprocessing summaries.
    parser.add_argument('--output_dir', type=str, default='./xxx/xx/processed_data', help='Directory for preprocessing outputs.')
    parser.add_argument('--analyze_only', action='store_true', help='Inspect the dataset without creating data loaders.')
    parser.add_argument('--test_only', action='store_true', help='Run the data loader check only.')

    args = parser.parse_args()

    # See README.md for project setup and required files.
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'data': {
                'dataset_root': args.dataset_root or './xxx/xx',
                'output_dir': args.output_dir or './test419gai_processed_data',
                'batch_size': 64
            }
        }

    # See README.md for project setup and required files.
    if args.dataset_root:
        config['data']['dataset_root'] = args.dataset_root
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir

    print("=" * 60)
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("=" * 60)
    print(f"Status: {config['data']['dataset_root']}")
    print(f"Status: {config['data']['output_dir']}")
    print(f"Status: {config['data'].get('batch_size', 8)}")
    print("=" * 60)

    # See README.md for project setup and required files.
    processor = ROIDataProcessor(
        dataset_root=config['data']['dataset_root']
    )

    # See README.md for project setup and required files.
    dataset_root = Path(config['data']['dataset_root'])
    if not dataset_root.exists():
        print(f"Status: Status: {dataset_root}")
        return

    required_dirs = ['train', 'val', 'test']
    required_subdirs = ['he', 'mpm']

    print("Processing...")
    for split_dir in required_dirs:
        split_path = dataset_root / split_dir
        if not split_path.exists():
            print(f"  Status: Status: {split_path}")
        else:
            for subdir in required_subdirs:
                subdir_path = split_path / subdir
                if not subdir_path.exists():
                    print(f"  Status: Status: {subdir_path}")
                else:
                    # See README.md for project setup and required files.
                    patient_folders = [d for d in subdir_path.iterdir() if d.is_dir()]
                    print(f"  {split_dir}/{subdir}: {len(patient_folders)} Status")

    # See README.md for project setup and required files.
    print("Processing...")
    test_filenames = [
        "9-2-C1+C6+C8.tif",  # See README.md for project setup and required files.
        "14-2-C1+C6.tif",  # See README.md for project setup and required files.
        "4-2-C6.tif",  # See README.md for project setup and required files.
        "10-2-C6.tif",  # See README.md for project setup and required files.
        "test.tif"  # See README.md for project setup and required files.
    ]

    for filename in test_filenames:
        tacs_types = processor.parse_tacs_type_from_filename(filename)
        print(f"  {filename} -> TACS: {tacs_types}")

    # See README.md for project setup and required files.
    print("Processing...")
    try:
        train_data, val_data, test_data = processor.create_all_datasets()
    except Exception as e:
        print(f"Status: {e}")
        return

    if args.analyze_only:
        print("Processing...")
        return

    # See README.md for project setup and required files.
    print("Processing...")
    metadata = processor.save_metadata_multi_label(train_data, val_data, test_data,
                                                   config['data']['output_dir'])

    if args.test_only:
        # See README.md for project setup and required files.
        print("Processing...")
        try:
            # See README.md for project setup and required files.
            train_dataset = PairedROIDataset(train_data, phase='train')
            val_dataset = PairedROIDataset(val_data, phase='val')
            test_dataset = PairedROIDataset(test_data, phase='test')

            print(f"  Status: {len(train_dataset)}")
            print(f"  Status: {len(val_dataset)}")
            print(f"  Status: {len(test_dataset)}")

            # See README.md for project setup and required files.
            print("Processing...")
            sample = train_dataset[0]
            print(f"  StatusID: {sample['patient_id']}")
            print(f"  Status: {sample['filename']}")
            print(f"  TACSStatus: {sample['tacs_type']}")
            if sample['he_image'] is not None:
                print(f"  HEStatus: {sample['he_image'].shape}")
                print(f"  HEStatus: {sample['he_size']}")
            if sample['mpm_image'] is not None:
                print(f"  MPMStatus: {sample['mpm_image'].shape}")
                print(f"  MPMStatus: {sample['mpm_size']}")

        except Exception as e:
            print(f"Status: {e}")
        return

    # See README.md for project setup and required files.
    print("Processing...")
    try:
        train_loader, val_loader, test_loader, metadata = create_data_loaders(config, verbose=True)

        # See README.md for project setup and required files.
        test_data_loading(train_loader, val_loader, test_loader)

    except Exception as e:
        print(f"Status: {e}")
        import traceback
        traceback.print_exc()
        print("Processing...")

    print("\n" + "=" * 60)
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("Processing...")
    print("=" * 60)


if __name__ == "__main__":
    main()
