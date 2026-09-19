import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

def get_transforms(split: str, target_size: Tuple[int, int] = (256, 256)) -> T.Compose:
    """
    FR-011: Strict Leakage Enforcement.
    Augmentations are strictly applied to the Training split ONLY.
    Validation and Test splits ONLY have Resize and Normalize applied.
    """
    if split == 'train':
        return T.Compose([
            T.ToImage(),
            T.Resize(size=target_size, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.3),
            T.RandomApply([T.ElasticTransform(alpha=50.0)], p=0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToImage(),
            T.Resize(size=target_size, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def get_seg_transforms(split: str, target_size: Tuple[int, int] = (256, 256)) -> T.Compose:
    """Segmentation-specific transforms with paired image+mask augmentation."""
    if split == 'train':
        return T.Compose([
            T.ToImage(),
            T.Resize(size=target_size, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.3),
            T.RandomApply([T.ElasticTransform(alpha=50.0)], p=0.2),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToImage(),
            T.Resize(size=target_size, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def split_dataset(data: List[Dict[str, Any]], stratify_col: str = 'class', random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    FR-011: Split data into Train (70%), Validation (15%), and Test (15%) using stratified splitting.
    """
    stratify_labels = [d.get(stratify_col) for d in data] if (len(data) > 0 and stratify_col in data[0]) else None
    
    # First split: Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        data, test_size=0.30, random_state=random_state, stratify=stratify_labels
    )
    
    temp_labels = [d.get(stratify_col) for d in temp_df] if (len(temp_df) > 0 and stratify_col in temp_df[0]) else None
    # Second split: Validation (15%) and Test (15%) from Temp (50% of 30%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, stratify=temp_labels
    )
    
    logger.info(f"Dataset split completed. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

class BaseMRIDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], split: str, target_size: Tuple[int, int] = (256, 256)):
        self.data = data
        self.split = split
        self.target_size = target_size
        self.transform = get_transforms(split, target_size)
        
    def _load_image(self, path: str) -> np.ndarray:
        """
        FR-007, FR-008: Load image and convert grayscale to 3-channel.
        Handles both absolute paths (from metadata) and relative project paths.
        """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            # Fallback: try to resolve as relative from cwd using just the
            # path suffix after known dataset roots
            import re
            for marker in ['data/brisc', 'data/pmram', 'data/cached_enhanced']:
                match = re.search(re.escape(marker).replace('/', '[/\\\\]') + r'.*', path)
                if match:
                    rel = match.group(0).replace('\\', '/')
                    image = cv2.imread(rel, cv2.IMREAD_COLOR)
                    if image is not None:
                        break
        if image is None:
            raise ValueError(f"Could not read image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class BRISCClassificationDataset(BaseMRIDataset):
    def __init__(self, data: List[Dict[str, Any]], split: str, class_to_idx: Dict[str, int], target_size: Tuple[int, int] = (256, 256)):
        super().__init__(data, split, target_size)
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int):
        item = self.data[idx]
        image_path = item['path']
        class_name = item['class']
        
        image = self._load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        return image, label

class BRISCSegmentationDataset(BaseMRIDataset):
    def __init__(self, data: List[Dict[str, Any]], split: str, target_size: Tuple[int, int] = (256, 256)):
        super().__init__(data, split, target_size)
        self.transform = get_seg_transforms(split, target_size)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int):
        item = self.data[idx]
        image_path = item['image_path']
        mask_path = item['mask_path']
        
        image = self._load_image(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Could not read mask at {mask_path}")
            
        # FR-009 & FR-010: rigorously verify dimensions match
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Dimension mismatch between image {image.shape[:2]} and mask {mask.shape[:2]}")
            
        # Ensure mask is binary (0 and 1)
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            image, mask = self.transform(image, tv_tensors.Mask(mask))
            
        # Add channel dimension to mask (1, H, W)
        mask = mask.unsqueeze(0) if getattr(mask, 'ndim', mask.dim()) == 2 else mask
        return image, mask

class PMRAMDataset(BaseMRIDataset):
    def __init__(self, data: List[Dict[str, Any]], split: str, target_size: Tuple[int, int] = (256, 256)):
        super().__init__(data, split, target_size)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx: int):
        item = self.data[idx]
        image_path = item['path']
        
        image = self._load_image(image_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, item['provenance']
