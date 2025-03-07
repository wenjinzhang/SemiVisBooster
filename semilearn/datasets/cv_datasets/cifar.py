# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
import torch
from PIL import ImageFilter

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]



class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class MultiViewTransform(object):

    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,
        focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- generate random views
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views

def make_transforms(
    name,
    rand_size=224,
    focal_size=96,
    rand_crop_scale=(0.3, 1.0),
    focal_crop_scale=(0.05, 0.3),
    color_jitter=1.0,
    rand_views=2,
    focal_views=10,
):
    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=color_jitter),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=color_jitter),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=rand_views,
        focal_views=focal_views
    )
    return transform


def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])
    if 'msn' in alg:
        msn_transform = make_transforms(
            name=name,
            rand_size=args.rand_size,
            focal_size=args.focal_size,
            rand_views=args.rand_views+1,
            focal_views=args.focal_views,)
    else:
        msn_transform = None

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
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
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_dset = BasicDataset(alg=alg,
                           data=lb_data, 
                           targets=lb_targets, 
                           num_classes=num_classes, 
                           transform=transform_weak, 
                           is_ulb=False,
                           msn_transform=msn_transform,
                           strong_transform=transform_strong,
                           onehot=False)

    ulb_dset = BasicDataset(alg=alg, 
                            data=ulb_data, 
                            targets=ulb_targets, 
                            num_classes=num_classes, 
                            transform=transform_weak, 
                            is_ulb=True,
                            msn_transform=msn_transform,
                            strong_transform=transform_strong,
                            onehot=False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg=alg,
                             data=test_data, 
                             targets=test_targets,
                             num_classes=num_classes, 
                             transform= transform_val, 
                             is_ulb=False,
                             msn_transform=msn_transform,
                             strong_transform=None,
                             onehot=False)

    return lb_dset, ulb_dset, eval_dset



cifar100_label_name = {0: 'apple', 1: 'aquarium fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle',9: 'bottle', 
10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle',
20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 
30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 
40: 'lamp', 41: 'lawn mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple tree', 48: 'motorcycle', 49: 'mountain', 
50: 'mouse', 51: 'mushroom', 52: 'oak tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm tree', 57: 'pear', 58: 'pickup truck', 59: 'pine tree', 
60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 
70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 
80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 
90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow tree', 97: 'wolf', 98: 'woman', 99: 'worm'}

cifar100_label_text = {
    0: "Apple: A round fruit with smooth, shiny skin, typically red, green, or yellow. Apples have a firm texture and are often depicted with a small stem on top.",
    1: "Aquarium Fish: Small, colorful fish with a variety of bright patterns and fins, often depicted swimming in water. The fish may have stripes, spots, or vibrant shades of blue, orange, and yellow.",
    2: "Baby: A young human with soft, round facial features and smooth skin. Babies are usually depicted wearing simple clothes or diapers, with a small, pudgy body and large eyes.",
    3: "Bear: A large, furry animal with a thick body and round ears. Bears are often depicted with brown or black fur, and their strong limbs and sharp claws are visible.",
    4: "Beaver: A medium-sized, brown rodent with a flat, paddle-shaped tail and large, prominent teeth. Beavers are often depicted in water or near dams they build with logs.",
    5: "Bed: A piece of furniture with a flat surface for sleeping, typically covered with sheets and pillows. Beds often have a rectangular shape and can have a headboard and footboard.",
    6: "Bee: A small flying insect with black and yellow stripes, transparent wings, and antennae. Bees are often shown collecting nectar from flowers or flying in the air.",
    7: "Beetle: A small insect with a hard, shiny exoskeleton, often depicted in black, red, or green. Beetles have six legs and a pair of wings hidden beneath their protective outer shell.",
    8: "Bicycle: A two-wheeled vehicle with a metal frame, pedals, and handlebars for steering. Bicycles often have thin tires and a seat for one rider, and are depicted in various colors.",
    9: "Bottle: A tall, cylindrical container with a narrow neck, often used to hold liquids. Bottles can be made of glass or plastic and may have a label on the front and a cap or lid on top.",
    10: "Bowl: A round, deep dish with a wide opening, often used for holding food like soup or cereal. Bowls can be made of ceramic, plastic, or metal and are depicted in various sizes and colors.",
    11: "Boy: A young male child with a slim body, often wearing casual clothing like shorts and t-shirts. Boys usually have short hair and may be playing or standing in a lively, energetic pose.",
    12: "Bridge: A structure that spans across a river, valley, or road, allowing vehicles and people to cross. Bridges can have a variety of shapes, including arches or flat decks, and may be made of metal or concrete.",
    13: "Bus: A large, rectangular vehicle with rows of windows and a bright color, typically yellow for school buses. Buses have multiple wheels and are designed to carry many passengers.",
    14: "Butterfly: A delicate insect with large, colorful wings that spread out in a symmetrical pattern. Butterflies have thin bodies and antennae, and their wings often display vibrant patterns.",
    15: "Camel: A large, desert-dwelling animal with a long neck and distinctive humps on its back. Camels are often depicted with brown fur and long legs, used for walking across sandy terrain.",
    16: "Can: A cylindrical container made of metal, used to hold liquids like soda or soup. Cans are usually depicted with a smooth surface and a label or brand on the front.",
    17: "Castle: A large, fortified building with high walls, towers, and battlements. Castles are often depicted as old stone structures, sometimes surrounded by a moat or situated on a hill.",
    18: "Caterpillar: A small, worm-like creature with a long body made of many segments. Caterpillars are often depicted with bright colors and tiny legs, and are shown crawling on leaves or branches.",
    19: "Cattle: Large farm animals with sturdy bodies, short fur, and visible horns. Cattle are often depicted grazing in fields, with shades of brown, black, or white fur.",
    20: "Chair: A piece of furniture with a flat seat and a backrest, designed for one person to sit on. Chairs often have four legs and may be made of wood, metal, or plastic.",
    21: "Chimpanzee: A small primate with black or brown fur, a rounded face, and long limbs. Chimpanzees are often depicted sitting or climbing, and their expressive faces resemble those of humans.",
    22: "Clock: A round or square device with a face showing numbers and moving hands to indicate time. Clocks can be analog with hour and minute hands, or digital with numbers on a screen.",
    23: "Cloud: A large, fluffy mass of condensed water vapor floating in the sky. Clouds are typically white or gray and come in various shapes, often depicted against a blue sky.",
    24: "Cockroach: A small, brown insect with a flat body, long antennae, and spindly legs. Cockroaches are often depicted scurrying across surfaces or hiding in dark places.",
    25: "Couch: A long, padded piece of furniture for sitting, typically with armrests and cushions. Couches are often depicted in living rooms, with fabric or leather upholstery in various colors.",
    26: "Crab: A small, hard-shelled sea creature with a flat, round body and ten legs, including two pincers. Crabs are often depicted walking sideways on the sand or near water.",
    27: "Crocodile: A large, reptilian predator with a long snout, sharp teeth, and tough, scaly skin. Crocodiles are often depicted in water or on riverbanks, with their powerful tails and armored bodies.",
    28: "Cup: A small, round container with a handle, used for drinking liquids like coffee or tea. Cups are typically made of ceramic or plastic and come in various colors and designs.",
    29: "Dinosaur: A large, prehistoric reptile with a long tail and sharp teeth. Dinosaurs are often depicted with scaly skin, large bodies, and clawed limbs, in shades of green or brown.",
    30: "Dolphin: A sleek, gray marine mammal with a streamlined body and a curved dorsal fin. Dolphins are often depicted swimming in the ocean, leaping out of the water in playful arcs.",
    31: "Elephant: A large, gray animal with thick skin, big ears, and a long trunk. Elephants are often depicted in herds, with their tusks and powerful legs visible.",
    32: "Flatfish: A fish with a flattened body and both eyes on one side of its head. Flatfish are often depicted with mottled brown or gray skin, lying on the ocean floor.",
    33: "Forest: A large area filled with trees, bushes, and wildlife. Forests are often depicted with dense green foliage, sunlight filtering through the leaves, and a variety of plants and animals.",
    34: "Fox: A small, agile animal with reddish-brown fur, pointed ears, and a bushy tail. Foxes are often depicted in woodland settings, with their sharp eyes and quick movements.",
    35: "Girl: A young female child, often depicted with long hair and wearing dresses or casual clothes. Girls are usually shown in playful or active poses, with cheerful expressions.",
    36: "Hamster: A small, furry rodent with a round body, short legs, and tiny paws. Hamsters are often depicted in cages or running on wheels, with soft fur in shades of brown or white.",
    37: "House: A building with a roof, windows, and doors, typically depicted as a place where people live. Houses come in various shapes and sizes, with details like chimneys, porches, and gardens.",
    38: "Kangaroo: A large, hopping marsupial with powerful hind legs, a long tail, and a pouch. Kangaroos are often depicted in grassy fields, leaping on their strong legs.",
    39: "Keyboard: A flat, rectangular device with rows of keys used for typing. Keyboards are often depicted with black or white keys, connected to computers or laptops.",
    40: "Lamp: A device with a bulb and a shade, used for lighting a room. Lamps come in various shapes, including table lamps and floor lamps, with a switch for turning them on and off.",
    41: "Lawn Mower: A machine with rotating blades used to cut grass. Lawn mowers are often depicted with large wheels, a handle for pushing, and a green or metal body.",
    42: "Leopard: A large, spotted cat with golden-yellow fur covered in black rosettes. Leopards are often depicted in trees or stalking prey, with their sleek, muscular bodies.",
    43: "Lion: A large, majestic cat with a golden mane (in males) and tawny fur. Lions are often depicted in pride settings, lounging in the savanna or roaring with power.",
    44: "Lizard: A small, scaly reptile with a long body and tail, often depicted in shades of green or brown. Lizards are shown crawling on rocks or walls, using their sharp claws for grip.",
    45: "Lobster: A large, hard-shelled sea creature with a long body, large claws, and a reddish-brown shell. Lobsters are often depicted underwater or on plates as a seafood dish.",
    46: "Man: An adult male with mature facial features, possibly including facial hair. He is usually seen wearing formal or casual adult clothing such as shirts, suits, or jackets. Men are taller and broader than boys, and may have short or neatly styled hair.",
    47: "Maple Tree: A large, deciduous tree with distinctive five-lobed leaves that turn bright red, orange, or yellow in the fall. The tree has a thick trunk and broad, spreading branches.",
    48: "Motorcycle: A two-wheeled vehicle with a sleek, metal frame and an engine, often depicted speeding down a road. Motorcycles come in various colors and are known for their loud engines.",
    49: "Mountain: A large, natural elevation of the earth's surface with steep sides, often depicted with rocky terrain and snowy peaks. Mountains are part of rugged landscapes, towering over valleys.",
    50: "Mouse: A small, gray or brown rodent with a long, thin tail and large ears. Mice are often depicted scurrying across the ground or nibbling on food.",
    51: "Mushroom: A small, umbrella-shaped fungus with a stalk and cap. Mushrooms come in a variety of colors, including white, brown, and red, and are often depicted growing in forests or fields.",
    52: "Oak Tree: A large tree with thick, sturdy branches and lobed leaves. Oak trees are often depicted with acorns scattered around their base and a broad, spreading canopy of leaves.",
    53: "Orange: A round, bright orange fruit with a thick, dimpled skin. Oranges are often depicted whole or sliced open to reveal juicy segments inside.",
    54: "Orchid: A delicate flower with soft, curved petals and vibrant colors like purple, pink, or white. Orchids are often depicted in elegant clusters, growing on long, thin stems.",
    55: "Otter: A small, playful aquatic mammal with a sleek, brown body and webbed feet. Otters are often depicted swimming on their backs or holding small objects with their paws.",
    56: "Palm Tree: A tall, tropical tree with a slender trunk and large, fan-shaped leaves. Palm trees are often depicted near beaches or in desert landscapes, with a distinctive silhouette.",
    57: "Pear: A bell-shaped fruit with smooth, green or yellow skin. Pears are often depicted whole or sliced to reveal a soft, juicy interior with small seeds in the center.",
    58: "Pickup Truck: A vehicle with a cab and an open cargo area in the back. Pickup trucks are often depicted on roads or farms, with a rugged, utilitarian design and large wheels.",
    59: "Pine Tree: A tall, evergreen tree with long, needle-like leaves and a thick trunk. Pine trees are often depicted with pinecones hanging from their branches, in forests or snowy landscapes.",
    60: "Plain: A flat, wide stretch of land with few trees, often covered in grass. Plains are depicted with open skies and vast horizons, sometimes dotted with wildflowers or grazing animals.",
    61: "Plate: A round, flat dish used for serving food. Plates come in various sizes and colors, often depicted with food neatly arranged on them.",
    62: "Poppy: A bright red flower with delicate, crinkled petals and a dark center. Poppies are often depicted growing in fields, with their vibrant color standing out against green foliage.",
    63: "Porcupine: A small, spiny mammal with a round body covered in sharp quills. Porcupines are often depicted in forests, with their quills raised as a defensive mechanism.",
    64: "Possum: A small, nocturnal marsupial with gray fur, a pointed snout, and a long, prehensile tail. Possums are often depicted in trees or foraging on the ground.",
    65: "Rabbit: A small, fluffy mammal with long ears, large eyes, and a twitching nose. Rabbits are often depicted hopping in fields or sitting upright, with soft fur in various colors.",
    66: "Raccoon: A small, nocturnal animal with gray fur, a bushy tail, and distinctive black markings around its eyes. Raccoons are often depicted scavenging for food or climbing trees.",
    67: "Ray: A flat, diamond-shaped fish with wide, wing-like fins and a long tail. Rays are often depicted swimming gracefully along the ocean floor, with their smooth, gray or brown skin.",
    68: "Road: A long, paved path for vehicles and people to travel on. Roads are often depicted winding through landscapes, with lanes marked by white or yellow lines.",
    69: "Rocket: A tall, cylindrical spacecraft with fins at the base, designed for launching into space. Rockets are often depicted blasting off with flames and smoke trailing behind them.",
    70: "Rose: A classic flower with soft, layered petals arranged in a spiral, often red, pink, yellow, or white. Roses are depicted with glossy green leaves and a thorny stem.",
    71: "Sea: A vast body of saltwater with waves and tides, often depicted in shades of blue or green. The sea is shown with its horizon blending into the sky, sometimes with boats or fish.",
    72: "Seal: A sleek, aquatic mammal with a streamlined body and flippers, often depicted lounging on rocks or swimming in the ocean. Seals have smooth, shiny fur and whiskers.",
    73: "Shark: A large, predatory fish with a sleek, gray body and sharp teeth. Sharks are often depicted swimming in the ocean with their dorsal fin visible above the water's surface.",
    74: "Shrew: A small, insect-eating mammal with a pointed snout, short fur, and tiny eyes. Shrews are often depicted scurrying along the ground, foraging for food.",
    75: "Skunk: A small, black and white mammal with a bushy tail and a distinctive stripe running down its back. Skunks are often depicted with their tail raised as a warning.",
    76: "Skyscraper: A tall, multi-story building made of glass and steel, often found in cities. Skyscrapers are depicted towering over other buildings, with shiny windows reflecting the sky.",
    77: "Snail: A small, slow-moving creature with a soft body and a coiled shell on its back. Snails are often depicted crawling on leaves or rocks, leaving a trail of slime behind.",
    78: "Snake: A long, slender reptile with smooth, scaly skin, often depicted coiled or slithering on the ground. Snakes come in a variety of colors and patterns.",
    79: "Spider: A small, eight-legged arachnid with a round body and long, spindly legs. Spiders are often depicted weaving webs or crawling on walls and leaves.",
    80: "Squirrel: A small, agile rodent with a bushy tail and sharp claws, often depicted climbing trees or gathering nuts. Squirrels have brown or gray fur and large, dark eyes.",
    81: "Streetcar: A long, narrow vehicle that runs on tracks, often found in urban areas. Streetcars are depicted with windows and doors for passengers, moving along a fixed path.",
    82: "Sunflower: A tall flower with a large, round center filled with seeds and bright yellow petals radiating outward. Sunflowers are depicted standing tall in fields, facing the sun.",
    83: "Sweet Pepper: A colorful, bell-shaped vegetable with smooth, shiny skin. Sweet peppers come in red, yellow, green, or orange, and are often depicted whole or sliced.",
    84: "Table: A flat, horizontal surface with legs, used for dining, working, or holding objects. Tables are often depicted in wood or metal, with rectangular or round shapes.",
    85: "Tank: A large, armored military vehicle with a rotating turret and a long barrel. Tanks are often depicted moving on tracks, with their large, heavy bodies visible in action.",
    86: "Telephone: A device with a handset and a base, used for communication. Telephones are often depicted with a keypad or rotary dial and a coiled cord connecting the parts.",
    87: "Television: A rectangular device with a screen used for watching shows and movies. Televisions are often depicted with a flat screen, remote control, and speakers.",
    88: "Tiger: A large, orange and black striped cat with a muscular body and a powerful build. Tigers are often depicted prowling through the jungle or lounging in the grass.",
    89: "Tractor: A large, powerful vehicle used for farming, with large rear wheels and a sturdy metal body. Tractors are often depicted plowing fields or pulling farm equipment.",
    90: "Train: A long, connected vehicle that runs on tracks, often used for transporting people or goods. Trains are depicted with multiple carriages and a locomotive at the front.",
    91: "Trout: A slender freshwater fish with smooth, speckled skin and a pointed head. Trout are often depicted swimming in clear streams, with shades of green or silver on their bodies.",
    92: "Tulip: A smooth, cup-shaped flower with tall, green stems and brightly colored petals. Tulips are depicted in shades of red, yellow, pink, or purple, often growing in neat rows.",
    93: "Turtle: A small, slow-moving reptile with a hard shell covering its body. Turtles are often depicted crawling on land or swimming in water, with their heads and legs sticking out from their shells.",
    94: "Wardrobe: A tall, rectangular piece of furniture with doors, used for storing clothes. Wardrobes are often depicted with wooden or mirrored doors and compartments for hanging or folding clothes.",
    95: "Whale: A large marine mammal with a streamlined body and a broad tail fin. Whales are often depicted swimming in the ocean, with their dark, smooth skin visible above the water.",
    96: "Willow Tree: A tall tree with long, drooping branches and narrow leaves. Willow trees are often depicted near water, with their graceful branches swaying in the breeze.",
    97: "Wolf: A large, wild canine with thick gray or brown fur, pointed ears, and sharp teeth. Wolves are often depicted in packs, howling at the moon or stalking prey.",
    98: "Woman: An adult female with mature features and a more defined body shape. She often wears dresses, skirts, or fashionable clothing, with longer hair that is styled in various ways. Women appear taller and more mature than girls.",
    99: "Worm: A small, long, and flexible invertebrate with no legs. Worms are often depicted wriggling through the soil, with their segmented bodies in shades of pink or brown."
}

# cifar100_label_text = {0: 'apple', 1: 'aquarium fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle',9: 'bottle', 
# 10: 'bowl', 
# 11: 'boy: A young male child with a slim body, often wearing casual clothing like shorts and t-shirts. Boys usually have short hair.', 
# 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle',
# 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 
# 30: 'dolphin： A marine mammal with a streamlined body and curved dorsal fin.', 
# 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 
# 35: 'girl: A young female child, often wearing dresses or casual clothes, with longer hair, sometimes tied in ponytails. Girls look smaller and more youthful.', 
# 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 
# 40: 'lamp', 41: 'lawn mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 
# 46: 'man: An adult male with mature facial features, sometimes facial hair. Often wears shirts, suits, or jackets. Taller and broader than boys.', 
# 47: 'maple tree: A large deciduous tree with five-lobed leaves, a thick trunk, and spreading branches.', 
# 48: 'motorcycle', 49: 'mountain', 
# 50: 'mouse', 51: 'mushroom', 52: 'oak tree', 53: 'orange', 
# 54: 'orchid: A delicate flower with curved petals and vibrant colors like purple, pink, or white. Grows in clusters on thin stems.', 
# 55: 'otter', 56: 'palm tree', 57: 'pear', 58: 'pickup truck', 
# 59: 'pine tree: A tall evergreen tree with needle-like leaves and a sturdy trunk. Produces pine cones and has a cone-shaped appearance.', 
# 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 
# 70: 'rose: A flower with layers of soft petals in red, pink, yellow, or white. Has a thorny stem and glossy leaves.', 
# 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 
# 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 
# 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow tree', 97: 'wolf', 
# 98: 'woman: An adult female with mature features, often wearing dresses or fashionable clothing. Typically has longer hair and looks taller than girls.', 
# 99: 'worm'}