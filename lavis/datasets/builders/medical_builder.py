# Added by Yang B.

import os
import warnings
import lavis.common.utils as utils

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.caption_datasets import (
    CaptionDataset,
    CaptionEvalDataset
)
from lavis.common.registry import registry


class MedicalBaseBuilder(BaseDatasetBuilder):
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        train_with_random_prompts = self.config.get('train_with_random_prompts', False)
        train_languages = self.config.get('train_languages', ['en'])
        zh_caption_key = self.config.get('zh_caption_key', None)
        concepts_path = self.config.get('concepts_path', None)
        n_concepts = self.config.get('n_concepts', 1000)

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
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                train_with_random_prompts=train_with_random_prompts,
                train_languages=train_languages,
                zh_caption_key=zh_caption_key,
                concepts_path=concepts_path,
                n_concepts=n_concepts,
            )

        return datasets


@registry.register_builder("iu_xray_caption")
class IUXrayCapBuilder(MedicalBaseBuilder):
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/iu_xray_caption.yaml",
    }


@registry.register_builder("mimic_cxr_caption")
class MIMICCXRCapBuilder(MedicalBaseBuilder):
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/mimic_cxr_caption.yaml",
    }


@registry.register_builder("clef_2023_caption")
class CLEF2023CapBuilder(MedicalBaseBuilder):
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/clef_2023_caption.yaml",
    }
