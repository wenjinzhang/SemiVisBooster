# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import pickle

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_imagenet(args, alg, name, num_labels = 0, percentage=0.01, data_dir='./data', include_lb_to_ulb=False):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])
    print("--------------------------> data_dir----", data_dir, name.lower())
    print("--------------------------> percentage:{}".format(percentage) )

    data_dir = os.path.join(data_dir, name.lower())

    dataset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, strong_transform=transform_strong)
    num_samples = len(dataset)
    num_class = len(dataset.classes)
    num_per_class = int((num_samples * args.percentage ) / num_class) + 1
    # dump label index
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join(base_dir, 'dataidx', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{int(args.percentage*100)}_{args.lb_imb_ratio}_seed{args.seed}_idx.pk')
    
    if os.path.exists(lb_dump_path):
        # lb_idx = np.load(lb_dump_path)
        print("load lb_idx from---->", lb_dump_path)
        with open(lb_dump_path, 'rb') as file:
            lb_idx = pickle.load(file)
    else:
        print("generate lb_idx ---->")
        lb_idx = None

    lb_dset = ImagenetDataset(root=os.path.join(data_dir, "train"), num_per_class=num_per_class ,transform=transform_weak, ulb=False, alg=alg, percentage=percentage, lb_index=lb_idx)

    if not os.path.exists(lb_dump_path):
        with open(lb_dump_path, 'wb') as file:
            print("save lb_idx ---->")
            pickle.dump(lb_dset.lb_idx, file)
    print("----->is loaded same as current:", lb_idx == lb_dset.lb_idx)
    
    ulb_dset = ImagenetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, alg=alg, ulb=True, strong_transform=transform_strong, include_lb_to_ulb=include_lb_to_ulb, lb_index=lb_dset.lb_idx)

    eval_dset = ImagenetDataset(root=os.path.join(data_dir, "val"), transform=transform_val, alg=alg, ulb=False)

    return lb_dset, ulb_dset, eval_dset
    


class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, strong_transform=None, percentage=-1, num_per_class = 0, include_lb_to_ulb=True, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        self.percentage = percentage
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index
        self.num_per_class = num_per_class

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"


    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)
        
        if self.lb_index is None:
            lb_idx = {}
        else:
            # load previous data
            lb_idx = self.lb_index

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                if not self.is_ulb:
                    if target_class in lb_idx:
                        # load existing label_idx
                        fnames = lb_idx[target_class]
                        random.shuffle(fnames)
                    else:
                        # generate label_idx
                        random.shuffle(fnames)
                        if self.percentage != -1:
                            sampel_num = len(fnames) if self.is_ulb else self.num_per_class
                            fnames = fnames[:sampel_num]
                            lb_idx[target_class] = fnames
                for fname in fnames:
                    if not self.include_lb_to_ulb:
                        if fname in self.lb_index[target_class]:
                            continue
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        gc.collect()
        self.lb_idx = lb_idx
        return instances

