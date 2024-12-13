import os
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.iu_xray_dataset import IUXrayDataset
import lavis.common.utils as utils
import warnings

@registry.register_builder("iu_xray")
class IUXrayBuilder(BaseDatasetBuilder):
    """
    Builder class for IU X-ray dataset.
    Properly inherits from BaseDatasetBuilder with minimal modifications.
    """
    train_dataset_cls = IUXrayDataset
    eval_dataset_cls = IUXrayDataset
    test_dataset_cls = IUXrayDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/iu_xray/defaults.yaml",
    }
    
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info
        
        
        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)
        
        # Get compressed features path from config
        compressed_features_path = None
        if hasattr(build_info, 'compressed_features'):
            compressed_features_path = build_info.compressed_features.storage
            if not os.path.isabs(compressed_features_path):
                compressed_features_path = utils.get_cache_path(compressed_features_path)
            
            if not os.path.exists(compressed_features_path):
                warnings.warn(f"Compressed features path {compressed_features_path} does not exist.")
                
        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue
            
            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # visual data storage path
            vis_path = vis_info.storage

            if not os.path.isabs(vis_path):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                vis_path = utils.get_cache_path(vis_path)

            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            # dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            if split == "train":
                dataset_cls = self.train_dataset_cls
            elif split == "val":
                dataset_cls = self.eval_dataset_cls
            else:  # split == "test"
                dataset_cls = self.test_dataset_cls
                
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                split= split,
                compressed_features_path=compressed_features_path
            )
        
        return datasets
    
    def _download_ann(self):
        """Skip download for IU X-ray as data should be prepared separately."""
        pass

    def _download_vis(self):
        """Skip download for IU X-ray as data should be prepared separately."""
        pass
