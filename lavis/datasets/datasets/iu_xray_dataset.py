import torch
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from lavis.common.registry import registry
import h5py
import logging
class IUXrayDataset(Dataset):
    """
    Dataset class for IU X-ray images and reports.
    Handles loading and processing of image pairs and their corresponding reports.
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split, compressed_features_path=None):
        """
        Initialize the IU X-ray dataset.

        Args:
            vis_processor: Processor for image preprocessing
            text_processor: Processor for text preprocessing
            vis_root: Root directory for images
            ann_paths: Path(s) to annotation file(s) 
            split: Dataset split ('train', 'val', or 'test')
            compressed_features_path: Path to compressed features h5 file
        """
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.image_dir = vis_root  # keep internal variable name consistent
        self.use_compressed_features = compressed_features_path is not None

        # Load compressed features if provided because we want to use them instead of raw images
        if compressed_features_path is not None:
            with h5py.File(compressed_features_path, 'r') as f:
                self.cluster_centers = torch.from_numpy(f['cluster_centers'][()]).float()  # (512, 768)
            logging.info(f"Loaded cluster centers with shape {self.cluster_centers.shape}")
        else:
            self.cluster_centers = None

        print(f"Loading {split} dataset from {ann_paths}")
        # Handle both single path and list of paths
        ann_path = ann_paths[0] if isinstance(ann_paths, (list, tuple)) else ann_paths
        
        # Load annotations for the specified split
        with open(ann_path, 'r', encoding='utf-8') as f:
            if split == 'eval':
                split = 'val'
            self.annotations = json.load(f)[split]
        print(f"Loaded {len(self.annotations)} samples for {split}")

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to fetch

        Returns:
            dict: Contains:
                - image: Tensor of stacked processed image pair [2, C, H, W]
                - text_input: Processed report text
                - study_id: Unique identifier for the study
        """
        # Get annotation for the current index
        ann = self.annotations[index]
        
        # Load and process the frontal and lateral view images
        image_0_path = os.path.join(self.image_dir, ann['image_path'][0])
        image_1_path = os.path.join(self.image_dir, ann['image_path'][1])
        
        # Open and convert images to RGB
        image_0 = Image.open(image_0_path).convert('RGB')
        image_1 = Image.open(image_1_path).convert('RGB')
        
        # Apply visual processing to both images
        image_0 = self.vis_processor(image_0)
        image_1 = self.vis_processor(image_1)

        # Process the report text
        report = self.text_processor(ann['report'])

        return {
            #    "image": torch.stack([image_0, image_1], dim=0),  # stack two views: [2, C, H, W]
            "image": image_0,
            "text_input": report,
            "study_id": ann['id'],
            "cluster_centers": self.cluster_centers  # (512, 768)
        }

    def collater(self, samples):
        """
        Collate a list of samples into a batch.

        Args:
            samples (list): List of samples from __getitem__

        Returns:
            dict: Batched samples with:
                - image: Stacked images [B, 2, C, H, W]
                - text_input: List or tensor of processed reports
                - study_id: List of study IDs
        """
        if len(samples) == 0:
            return {}

        # Create batch dictionary
        batch = {
            key: [sample[key] for sample in samples] for key in samples[0].keys()
        }

        # Stack images if present
        if "image" in batch:
            images = batch["image"]
            if torch.is_tensor(images[0]):
                images = torch.stack(images, dim=0)  # [B, 2, C, H, W]
            batch["image"] = images

        # Handle text inputs - stack if tensors, keep as list otherwise
        if "text_input" in batch:
            if torch.is_tensor(batch["text_input"][0]):
                batch["text_input"] = torch.stack(batch["text_input"])
                
        # All samples share the same cluster centers because they represent the whole dataset
        if "cluster_centers" in batch:
            batch["cluster_centers"] = batch["cluster_centers"][0]  
        return batch

