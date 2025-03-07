# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
from PIL import Image

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


def get_flowers102(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, 'Flowers102')
    dset = dset(data_dir, split='train', download=True)
    data, targets = dset._image_files, dset._labels
    

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(int(math.floor(img_size / crop_ratio))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = Flowers102Dataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

    ulb_dset = Flowers102Dataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, 'Flowers102')
    dset = dset(data_dir, split='test', download=True)
    test_data, test_targets = dset._image_files, dset._labels
    eval_dset = Flowers102Dataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset


class Flowers102Dataset(BasicDataset):
    def __sample__(self, idx):
        path = self.data[idx]
        img = Image.open(path).convert("RGB")
        target = self.targets[idx]
        return img, target 


flowers102_label_name = {
    0: "pink primrose",
    1: "hard-leaved pocket orchid",
    2: "canterbury bells",
    3: "sweet pea",
    4: "english marigold",
    5: "tiger lily",
    6: "moon orchid",
    7: "bird of paradise",
    8: "monkshood",
    9: "globe thistle",
    10: "snapdragon",
    11: "colt's foot",
    12: "king protea",
    13: "spear thistle",
    14: "yellow iris",
    15: "globe-flower",
    16: "purple coneflower",
    17: "peruvian lily",
    18: "balloon flower",
    19: "giant white arum lily",
    20: "fire lily",
    21: "pincushion flower",
    22: "fritillary",
    23: "red ginger",
    24: "grape hyacinth",
    25: "corn poppy",
    26: "prince of wales feathers",
    27: "stemless gentian",
    28: "artichoke",
    29: "sweet william",
    30: "carnation",
    31: "garden phlox",
    32: "love in the mist",
    33: "mexican aster",
    34: "alpine sea holly",
    35: "ruby-lipped cattleya",
    36: "cape flower",
    37: "great masterwort",
    38: "siam tulip",
    39: "lenten rose",
    40: "barbeton daisy",
    41: "daffodil",
    42: "sword lily",
    43: "poinsettia",
    44: "bolero deep blue",
    45: "wallflower",
    46: "marigold",
    47: "buttercup",
    48: "oxeye daisy",
    49: "common dandelion",
    50: "petunia",
    51: "wild pansy",
    52: "primula",
    53: "sunflower",
    54: "pelargonium",
    55: "bishop of llandaff",
    56: "gaura",
    57: "geranium",
    58: "orange dahlia",
    59: "pink-yellow dahlia",
    60: "cautleya spicata",
    61: "japanese anemone",
    62: "black-eyed susan",
    63: "silverbush",
    64: "californian poppy",
    65: "osteospermum",
    66: "spring crocus",
    67: "iris",
    68: "water lily",
    69: "rose",
    70: "thorn apple",
    71: "morning glory",
    72: "passion flower",
    73: "lotus",
    74: "toad lily",
    75: "anthurium",
    76: "frangipani",
    77: "clematis",
    78: "hibiscus",
    79: "columbine",
    80: "desert-rose",
    81: "tree mallow",
    82: "magnolia",
    83: "cyclamen",
    84: "watercress",
    85: "canna lily",
    86: "hippeastrum",
    87: "bee balm",
    88: "air plant",
    89: "foxglove",
    90: "bougainvillea",
    91: "camellia",
    92: "mallow",
    93: "mexican petunia",
    94: "bromelia",
    95: "blanket flower",
    96: "trumpet creeper",
    97: "blackberry lily",
    98: "common tulip",
    99: "wild rose"
}