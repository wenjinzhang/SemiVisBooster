# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math 
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.utils import sample_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment


mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
img_size = 96

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(img_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_stl10(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=False):
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset_lb = dset(data_dir, split='train', download=True)
    dset_ulb = dset(data_dir, split='unlabeled', download=True)
    lb_data, lb_targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])

    # Note this data can have imbalanced labeled set, and with unknown unlabeled set
    ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
    lb_idx, _ = sample_labeled_unlabeled_data(args, lb_data, lb_targets, num_classes,
                                              lb_num_labels=num_labels,
                                              ulb_num_labels=args.ulb_num_labels,
                                              lb_imbalance_ratio=args.lb_imb_ratio,
                                              ulb_imbalance_ratio=args.ulb_imb_ratio,
                                              load_exist=True)
    ulb_targets = np.ones((ulb_data.shape[0], )) * -1
    lb_data, lb_targets = lb_data[lb_idx], lb_targets[lb_idx]
    if include_lb_to_ulb:
        ulb_data = np.concatenate([lb_data, ulb_data], axis=0)
        ulb_targets = np.concatenate([lb_targets, np.ones((ulb_data.shape[0] - lb_data.shape[0], )) * -1], axis=0)
    ulb_targets = ulb_targets.astype(np.int64)

    # output the distribution of labeled data for remixmatch
    count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        count[c] += 1
    dist = np.array(count, dtype=float)
    dist = dist / dist.sum()
    dist = dist.tolist()
    out = {"distribution": dist}
    output_file = r"./data_statistics/"
    output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)
    with open(output_path, 'w') as w:
        json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, transform_strong, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset_lb = dset(data_dir, split='test', download=True)
    data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset


stl10_label_name = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck"
}

stl10_label_text = {
    0: "Airplane: A sleek, metallic body with wings extended on both sides. Often shown flying in the sky with clouds in the background. The fuselage may have windows and landing gear visible, depending on the angle.",
    1: "Bird: A small animal with feathers, typically perched on a branch or flying. Wings are visible in various positions, with a beak and distinct head shape. Backgrounds may include trees, sky, or grassy areas.",
    2: "Car: A four-wheeled vehicle with a shiny exterior, sometimes reflecting light. Shapes can vary from compact to larger cars, with visible tires, windows, and headlights. The car often appears on roads or in parking lots.",
    3: "Cat: A furry, domesticated animal with pointy ears, whiskers, and a long tail. Typically shown sitting or walking, with various fur colors and patterns. The face features sharp eyes and a small nose.",
    4: "Deer: A large animal with a slender body and long legs. Usually shown in a natural outdoor setting like a forest. It has a graceful appearance with antlers (in males) and a short tail.",
    5: "Dog: A furry, domesticated animal with floppy or pointy ears and a wagging tail. Shown in various poses, often with a friendly expression. Fur texture and color vary, and the eyes and snout are prominent features.",
    6: "Horse: A large, muscular animal with a long neck, flowing mane, and tail. Typically depicted running or standing in open fields or grassy areas. The legs and hooves are sturdy, and the head features prominent eyes and ears.",
    7: "Monkey: A small to medium-sized animal with a tail and expressive face. Often shown in playful poses, sometimes climbing trees or hanging from branches. It has a distinctive face with round eyes and a visible nose.",
    8: "Ship: A large watercraft with a hull, often seen floating on water. It may have sails or a streamlined body with decks visible. Surrounding water and waves are common visual elements, with occasional visible lifeboats or chimneys.",
    9: "Truck: A large vehicle with an extended cab and cargo area in the back. The truck's wheels are big and sturdy, with a prominent front grill and headlights. It is often shown on roads or in industrial settings."
}