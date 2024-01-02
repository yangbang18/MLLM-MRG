# Added by Yang B.

import os
import torch
import random
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


PROMPTS = {
    'en_default': '',
    'en': [
        '',
        'answer',
        'English output',
        'describe the image',
        'describe the image in English',
        'tell me your impressions of this image',
        'tell me your findings of this image',
        '英文答案'
        '英文输出'
        '用英文描述这幅图',
        '用英文告诉我你对这张图的印象',
        '你对这张图的调查发现是什么？用英文回答',
    ],
    'zh_default': '描述这幅图',
    'zh': [
        '中文输出',
        '描述这幅图',
        '用中文描述这幅图',
        '告诉我你对这张图的印象',
        '告诉我你对这张图的印象，用中文回答',
        '你对这张图的调查发现是什么？',
        'Chinese answer',
        'Chinese output',
        'describe the image in Chinese',
        'tell me your impressions of this image in Chinese',
        'tell me your findings of this image in Chinese',
    ],
}


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class CaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 train_with_random_prompts=False,
                 train_languages=['en'],
                 zh_caption_key=None,
                 concepts_path=None,
                 n_concepts=1000,
                 ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.train_with_random_prompts = train_with_random_prompts
        self.train_languages = train_languages
        self.zh_caption_key = zh_caption_key
        self.n_concepts = n_concepts
        if concepts_path is not None:
            print('Load concepts from', concepts_path)
            self.concepts = open(concepts_path, 'r').read().strip().split('\n')[:self.n_concepts]
            print(f'There are {len(self.concepts)} concepts')
            assert len(self.concepts) == self.n_concepts

            self.concepts_label_cache = {}

    def __getitem__(self, index):

        lang = random.choice(self.train_languages)
        assert lang in ['en', 'zh']

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        if lang == 'en':
            caption = ann["caption"]
        else:
            assert 'zh_caption' in ann
            assert type(ann['zh_caption']) is dict
            zh_caption_key = self.zh_caption_key or random.choice(list(ann['zh_caption'].keys()))
            caption = ann['zh_caption'][zh_caption_key]

        caption = self.text_processor(caption)

        out = {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
        
        if self.train_with_random_prompts:
            out['input_prompt'] = random.choice(PROMPTS[lang])
        
        if hasattr(self, 'concepts'):
            # construct labels for concept detection
            assert lang == 'en'
            if index not in self.concepts_label_cache:
                label = torch.zeros(self.n_concepts)
                for i, concept in enumerate(self.concepts):
                    if concept in caption:
                        label[i] = 1
                self.concepts_label_cache[index] = label
            else:
                label = self.concepts_label_cache[index]
            
            out['concepts_label'] = label
        
        return out


class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 train_with_random_prompts=False,
                 **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.train_with_random_prompts = train_with_random_prompts

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        out = {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }
        
        if self.train_with_random_prompts:
            out['input_prompt'] = PROMPTS['en_default']
        
        return out
