# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import math
from torchvision.datasets import folder as dataset_parser
from torchvision.transforms import transforms
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from .datasetbase import BasicDataset


def get_semi_aves(args, alg, dataset, train_split='l_train_val', ulb_split='u_train_in', data_dir='./data'):
    assert train_split in ['l_train', 'l_train_val']

    data_dir = os.path.join(data_dir, 'semi_fgvc')

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
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
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

    # NOTE this dataset is inherently imbalanced with unknown distribution
    train_labeled_dataset = iNatDataset(alg, data_dir, train_split, dataset, transform=transform_weak, transform_strong=transform_strong)
    train_unlabeled_dataset = iNatDataset(alg, data_dir, ulb_split, dataset, is_ulb=True, transform=transform_weak, transform_strong=transform_strong)
    test_dataset = iNatDataset(alg, data_dir, 'test', dataset, transform=transform_val)

    num_data_per_cls = [0] * train_labeled_dataset.num_classes
    for l in train_labeled_dataset.targets:
        num_data_per_cls[l] += 1

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def make_dataset(dataset_root, split, task='All', pl_list=None):
    split_file_path = os.path.join(dataset_root, task, split + '.txt')

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    ## Use PL + l_train
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        # elif task[:9] == 'semi_aves':
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        img += pl_list

    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            img[idx][0] = os.path.join(dataset_root, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]
    num_classes = len(set(classes))
    print('# images in {}: {}'.format(split, len(img)))
    return img, num_classes, classes


class iNatDataset(BasicDataset):
    def __init__(self, alg, dataset_root, split, task='All', transform=None, transform_strong=None,
                 loader=dataset_parser.default_loader, pl_list=None, is_ulb=False):

        self.alg = alg
        self.is_ulb = is_ulb
        self.loader = loader
        self.dataset_root = dataset_root
        self.task = task

        self.samples, self.num_classes, self.targets = make_dataset(self.dataset_root, split, self.task, pl_list=pl_list)

        self.transform = transform
        self.strong_transform = transform_strong
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"

        self.data = []
        for i in range(len(self.samples)):
            self.data.append(self.samples[i][0])
    
    def __sample__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        return img, target 


    def __len__(self):
        return len(self.data)
    


semi_aves_label_name = {
    0: "Charadrius semipalmatus",
    1: "Tachycineta thalassina",
    2: "Stelgidopteryx serripennis",
    3: "Cassiculus melanicterus",
    4: "Jacana spinosa",
    5: "Haematopus bachmani",
    6: "Phoenicopterus ruber",
    7: "Phoenicurus phoenicurus",
    8: "Pelecanus occidentalis",
    9: "Archilochus alexandri",
    10: "Pelecanus conspicillatus",
    11: "Falco tinnunculus",
    12: "Tachybaptus novaehollandiae",
    13: "Petrochelidon pyrrhonota",
    14: "Falco peregrinus",
    15: "Oenanthe oenanthe",
    16: "Calidris pusilla",
    17: "Calidris alpina",
    18: "Streptopelia chinensis",
    19: "Limosa fedoa",
    20: "Anthochaera carunculata",
    21: "Columba livia",
    22: "Phalaropus lobatus",
    23: "Thalasseus sandvicensis",
    24: "Larus canus",
    25: "Motacilla cinerea",
    26: "Aechmophorus occidentalis",
    27: "Larus glaucescens",
    28: "Dendrocopos major",
    29: "Rissa tridactyla",
    30: "Molothrus aeneus",
    31: "Entomyzon cyanotis",
    32: "Alopochen aegyptiaca",
    33: "Lonchura punctulata",
    34: "Anas platyrhynchos",
    35: "Aythya australis",
    36: "Dicrurus macrocercus",
    37: "Anser albifrons",
    38: "Buteo regalis",
    39: "Haliaeetus leucocephalus",
    40: "Morus bassanus",
    41: "Certhia familiaris",
    42: "Rostrhamus sociabilis",
    43: "Cracticus torquatus",
    44: "Passerina amoena",
    45: "Aythya fuligula",
    46: "Chondestes grammacus",
    47: "Fulica atra",
    48: "Mergus merganser",
    49: "Egretta thula",
    50: "Pyrrhula pyrrhula",
    51: "Aquila chrysaetos",
    52: "Setophaga townsendi",
    53: "Cygnus atratus",
    54: "Cygnus buccinator",
    55: "Branta canadensis",
    56: "Circus hudsonius",
    57: "Anhinga anhinga",
    58: "Buteo jamaicensis",
    59: "Pica pica",
    60: "Corvus frugilegus",
    61: "Bonasa umbellus",
    62: "Myiozetetes similis",
    63: "Progne subis",
    64: "Megaceryle alcyon",
    65: "Molothrus ater",
    66: "Piranga rubra",
    67: "Vanellus chilensis",
    68: "Cygnus olor",
    69: "Passerculus sandwichensis",
    70: "Chroicocephalus philadelphia",
    71: "Setophaga palmarum",
    72: "Himantopus himantopus",
    73: "Setophaga occidentalis",
    74: "Buteo lineatus",
    75: "Zonotrichia leucophrys",
    76: "Saltator coerulescens",
    77: "Tyrannus vociferans",
    78: "Junco hyemalis",
    79: "Pitangus sulphuratus",
    80: "Sylvia communis",
    81: "Melanerpes erythrocephalus",
    82: "Rhipidura leucophrys",
    83: "Quiscalus major",
    84: "Sphyrapicus varius",
    85: "Sitta europaea",
    86: "Cyanocorax yncas",
    87: "Corvus brachyrhynchos",
    88: "Sitta canadensis",
    89: "Cyanistes caeruleus",
    90: "Colaptes auratus",
    91: "Baeolophus inornatus",
    92: "Erithacus rubecula",
    93: "Poecile carolinensis",
    94: "Passer domesticus",
    95: "Sphecotheres vieilloti",
    96: "Mniotilta varia",
    97: "Dryocopus pileatus",
    98: "Prunella modularis",
    99: "Megarynchus pitangua",
    100: "Melospiza georgiana",
    101: "Agelaius phoeniceus",
    102: "Lanius cristatus",
    103: "Catharus ustulatus",
    104: "Ammodramus savannarum",
    105: "Thryothorus ludovicianus",
    106: "Troglodytes aedon",
    107: "Setophaga caerulescens",
    108: "Neochmia temporalis",
    109: "Setophaga citrina",
    110: "Myiodynastes luteiventris",
    111: "Setophaga discolor",
    112: "Pipilo erythrophthalmus",
    113: "Pica nuttalli",
    114: "Toxostoma longirostre",
    115: "Ardea melanocephala",
    116: "Microcarbo africanus",
    117: "Charadrius wilsonia",
    118: "Spizella breweri",
    119: "Patagioenas flavirostris",
    120: "Chlidonias hybrida",
    121: "Sterna paradisaea",
    122: "Calidris virgata",
    123: "Acridotheres cristatellus",
    124: "Tigrisoma lineatum",
    125: "Ciconia nigra",
    126: "Oriolus chinensis",
    127: "Chloroceryle amazona",
    128: "Sylvia curruca",
    129: "Pernis ptilorhynchus",
    130: "Charadrius alexandrinus",
    131: "Certhia brachydactyla",
    132: "Myiodynastes maculatus",
    133: "Larus pacificus",
    134: "Treron vernans",
    135: "Urocissa erythroryncha",
    136: "Microcarbo pygmeus",
    137: "Elseyornis melanops",
    138: "Mycteria leucocephala",
    139: "Gerygone igata",
    140: "Fulica cristata",
    141: "Turdus leucomelas",
    142: "Brotogeris jugularis",
    143: "Pycnonotus aurigaster",
    144: "Lagonosticta senegala",
    145: "Poecile hudsonicus",
    146: "Ploceus velatus",
    147: "Ploceus cucullatus",
    148: "Plocepasser mahali",
    149: "Calidris ruficollis",
    150: "Tringa stagnatilis",
    151: "Empidonax virescens",
    152: "Polemaetus bellicosus",
    153: "Lamprotornis superbus",
    154: "Jynx torquilla",
    155: "Buteo rufinus",
    156: "Cecropis daurica",
    157: "Anser fabalis",
    158: "Coscoroba coscoroba",
    159: "Anthus petrosus",
    160: "Torgos tracheliotos",
    161: "Lamprotornis nitens",
    162: "Habia fuscicauda",
    163: "Calonectris diomedea",
    164: "Vanellus senegallus",
    165: "Polioptila californica",
    166: "Cinnyris asiaticus",
    167: "Saltator aurantiirostris",
    168: "Burhinus vermiculatus",
    169: "Charadrius collaris",
    170: "Cyanocorax beecheii",
    171: "Myiozetetes cayanensis",
    172: "Dicrurus leucophaeus",
    173: "Phoeniculus purpureus",
    174: "Colaptes chrysoides",
    175: "Lophaetus occipitalis",
    176: "Ixobrychus minutus",
    177: "Ardea goliath",
    178: "Stercorarius longicaudus",
    179: "Aethopyga siparaja",
    180: "Prinia subflava",
    181: "Fluvicola nengeta",
    182: "Lophoceros alboterminatus",
    183: "Vireo atricapilla",
    184: "Halcyon senegalensis",
    185: "Stelgidopteryx ruficollis",
    186: "Ortalis cinereiceps",
    187: "Forpus conspicillatus",
    188: "Trogon rufus",
    189: "Anthracothorax nigricollis",
    190: "Rhipidura javanica",
    191: "Dendrocopos syriacus",
    192: "Eucometis penicillata",
    193: "Poecile sclateri",
    194: "Emberiza rustica",
    195: "Colaptes punctigula",
    196: "Ocreatus underwoodii",
    197: "Prionops plumatus",
    198: "Antrostomus vociferus",
    199: "Spheniscus magellanicus"
}



semi_aves_label_text = {
    0: "Charadrius semipalmatus (Semipalmated Plover): Small shorebird with brown back, white belly, single black neck band, white eyebrow stripe, and black mask. Orange legs and short orange bill with black tip.",
    1: "Tachycineta thalassina (Violet-green Swallow): Vibrant green back with violet wings and white belly. Square tail and white rump patches visible in flight.",
    2: "Stelgidopteryx serripennis (Northern Rough-winged Swallow): Dull brown above and light brown below with no bold markings. Narrow, pointed wings and tail, often seen close to water.",
    3: "Cassiculus melanicterus (Yellow-winged Cacique): Glossy black with bright yellow wing and tail feathers. Black beak and sharp yellow-black contrast in open areas.",
    4: "Jacana spinosa (Northern Jacana): Slim, long-legged bird with extended toes for walking on water plants. Brown body, black neck and head, yellow shield, and beak.",
    5: "Haematopus bachmani (Black Oystercatcher): Large shorebird with all-black feathers, bright orange-red bill, and yellow eye ring, typically on rocky shores.",
    6: "Phoenicopterus ruber (American Flamingo): Tall, pinkish-orange with slender legs and a distinctive black-tipped, downward-curving bill.",
    7: "Phoenicurus phoenicurus (Common Redstart): Small, with an orange chest, black face, slate-gray back, and reddish-orange tail feathers.",
    8: "Pelecanus occidentalis (Brown Pelican): Large, with a brown body, pale yellow head, and long, scooped bill. Known for its broad wingspan.",
    9: "Archilochus alexandri (Black-chinned Hummingbird): Tiny with metallic green feathers, white chest, and dark chin. Males flash a purple collar in sunlight.",
    10: "Pelecanus conspicillatus (Australian Pelican): Large, white body, pinkish bill with a yellow pouch, and black wingtips.",
    11: "Falco tinnunculus (Common Kestrel): Sleek raptor with a rusty-brown body, black spots, and long tail with a white tip, known for hovering in flight.",
    12: "Tachybaptus novaehollandiae (Australasian Grebe): Small with black upperparts and a distinct golden-yellow stripe near the eye. Breeding adults have a bright chestnut patch.",
    13: "Petrochelidon pyrrhonota (Cliff Swallow): Compact, with a pale, buffy rump, square tail, and white forehead contrasting with its darker head.",
    14: "Falco peregrinus (Peregrine Falcon): Compact with bluish-gray wings, dark crown, and a barred chest. Known for speed in flight.",
    15: "Oenanthe oenanthe (Northern Wheatear): Small with pale gray upperparts, white underparts, and a black stripe from bill to eye. Tail has a distinctive white patch.",
    16: "Calidris pusilla (Semipalmated Sandpiper): Small, with grayish-brown plumage, short legs, and a short, black bill. Often found in flocks on mudflats.",
    17: "Calidris alpina (Dunlin): Medium-sized shorebird with reddish-brown back and a black patch on the belly during breeding season.",
    18: "Streptopelia chinensis (Spotted Dove): Light brown with a distinctive black-and-white 'necklace' on its nape and spotted wings.",
    19: "Limosa fedoa (Marbled Godwit): Large shorebird with long legs, a long upturned bill, and a marbled brown plumage.",
    20: "Anthochaera carunculata (Red Wattlebird): Grayish-brown with streaked plumage, distinctive red wattles near its neck, and a white streak along the wing edges.",
    21: "Columba livia (Rock Pigeon): Common bird with blue-gray plumage, a white rump, and two black bands on the wings.",
    22: "Phalaropus lobatus (Red-necked Phalarope): Small shorebird with gray back, white belly, and red neck in breeding season. Unique spinning behavior while feeding.",
    23: "Thalasseus sandvicensis (Sandwich Tern): Medium-sized tern with a slender black bill tipped with yellow, white body, and a black cap.",
    24: "Larus canus (Common Gull): Medium-sized with a gray back, white underparts, yellow legs, and a slightly drooped bill.",
    25: "Motacilla cinerea (Grey Wagtail): Small, slim bird with a gray back, bright yellow underparts, and a long, constantly wagging tail.",
    26: "Aechmophorus occidentalis (Western Grebe): Black and white with a long neck, sharp red eye, and distinctive courtship displays.",
    27: "Larus glaucescens (Glaucous-winged Gull): Pale gray back, white head and underparts, and wingtips that are light gray, not black.",
    28: "Dendrocopos major (Great Spotted Woodpecker): Black and white with a bright red patch on the lower belly and, in males, a red spot on the nape.",
    29: "Rissa tridactyla (Black-legged Kittiwake): Medium-sized gull with pure white body, gray wings tipped with black, and black legs.",
    30: "Molothrus aeneus (Bronzed Cowbird): Glossy, bronzed feathers, stout body, and a short tail. Often found around cattle or in open fields.",
    31: "Entomyzon cyanotis (Blue-faced Honeyeater): Large honeyeater with blue facial skin, olive-green back, and white belly.",
    32: "Alopochen aegyptiaca (Egyptian Goose): Distinctive brown patch around the eye and chest, pale gray body, and pinkish legs.",
    33: "Lonchura punctulata (Scaly-breasted Munia): Small bird with a brown body and scaly pattern on the chest and belly.",
    34: "Anas platyrhynchos (Mallard): Classic duck with iridescent green head (in males), brown chest, and white-bordered blue wing patch.",
    35: "Aythya australis (Hardhead): Brown duck with a white undertail patch and reddish-brown eyes, commonly seen on inland lakes.",
    36: "Dicrurus macrocercus (Black Drongo): Glossy black with a deeply forked tail and aggressive territorial behavior.",
    37: "Anser albifrons (Greater White-fronted Goose): Brown body with white patch on the forehead and orange legs, seen in flocks.",
    38: "Buteo regalis (Ferruginous Hawk): Large hawk with rust-colored feathers, pale head, and broad wings. Known for soaring flight.",
    39: "Haliaeetus leucocephalus (Bald Eagle): Iconic raptor with a white head and tail, yellow beak, and dark brown body.",
    40: "Morus bassanus (Northern Gannet): Large seabird with white body, black wingtips, and long neck. Known for plunging dives.",
    41: "Certhia familiaris (Eurasian Treecreeper): Small, brown, with a slender downcurved bill, and streaked body. Climbs tree trunks in a spiral.",
    42: "Rostrhamus sociabilis (Snail Kite): Dark, slender hawk with a hooked beak for feeding on snails. Long wings and tail.",
    43: "Cracticus torquatus (Grey Butcherbird): Gray with a black head and distinctive hooked bill. Known for melodious calls.",
    44: "Passerina amoena (Lazuli Bunting): Small, bright blue head and back, with a rusty chest and white belly (in males).",
    45: "Aythya fuligula (Tufted Duck): Black and white duck with a distinct tuft on its head and golden eyes.",
    46: "Chondestes grammacus (Lark Sparrow): Brown with a striped face pattern of black, white, and chestnut, and a long tail.",
    47: "Fulica atra (Eurasian Coot): Black body, white frontal shield and bill, and lobed toes, often found on freshwater lakes.",
    48: "Mergus merganser (Common Merganser): Large, slender duck with green head (males), reddish bill, and white body.",
    49: "Egretta thula (Snowy Egret): White feathers, black legs, yellow feet, and slender black bill, seen in wetlands.",
    50: "Pyrrhula pyrrhula (Eurasian Bullfinch): Chunky with a black cap, grayish back, and a bright reddish chest (in males).",
    51: "Aquila chrysaetos (Golden Eagle): Dark brown raptor with golden nape feathers and a massive wingspan.",
    52: "Setophaga townsendi (Townsend's Warbler): Black face and throat with bright yellow cheeks, and streaked green and black back.",
    53: "Cygnus atratus (Black Swan): Large waterfowl with all-black feathers, red bill, and curled wing feathers.",
    54: "Cygnus buccinator (Trumpeter Swan): Largest swan with all-white feathers, long neck, and black bill.",
    55: "Branta canadensis (Canada Goose): Brown body, long black neck, and white chinstrap. Known for migratory 'V' formations.",
    56: "Circus hudsonius (Northern Harrier): Slim hawk with an owl-like face, white rump patch, and low flight over fields.",
    57: "Anhinga anhinga (Anhinga): Long neck, black body, and dagger-like bill. Often seen drying wings in the sun.",
    58: "Buteo jamaicensis (Red-tailed Hawk): Large, broad-winged raptor with a reddish tail and variable plumage patterns.",
    59: "Pica pica (Eurasian Magpie): Black and white body with iridescent blue-green wings and tail.",
    60: "Corvus frugilegus (Rook): Black, glossy plumage with a bare, pale bill base, seen in open fields.",
    61: "Bonasa umbellus (Ruffed Grouse): Medium-sized, brown bird with a fan-shaped tail and ruffs on neck feathers.",
    62: "Myiozetetes similis (Social Flycatcher): Bright yellow belly, brown back, black mask, and white eyebrow stripe.",
    63: "Progne subis (Purple Martin): Large swallow with glossy blue-black plumage (males) and forked tail.",
    64: "Megaceryle alcyon (Belted Kingfisher): Slate blue back, white belly, and distinctive shaggy crest. Known for hunting fish.",
    65: "Molothrus ater (Brown-headed Cowbird): Brown head, glossy black body (males), and known for brood parasitism.",
    66: "Piranga rubra (Summer Tanager): Bright red in males, yellowish in females, with a stout bill and seen in wooded areas.",
    67: "Vanellus chilensis (Southern Lapwing): Gray body with a black throat, pinkish legs, and a distinct crest.",
    68: "Cygnus olor (Mute Swan): Large, with a long neck, white body, and orange bill with a black knob.",
    69: "Passerculus sandwichensis (Savannah Sparrow): Streaked brown with a yellow patch above the eye and short tail.",
    70: "Chroicocephalus philadelphia (Bonaparte's Gull): Small gull with white body, black head in breeding season, and red legs.",
    71: "Setophaga palmarum (Palm Warbler): Yellowish underparts, brownish back, and rusty crown, often seen bobbing its tail.",
    72: "Himantopus himantopus (Black-winged Stilt): Slender shorebird with black wings, white body, and very long pink legs.",
    73: "Setophaga occidentalis (Hermit Warbler): Bright yellow head, gray back, and white belly with streaked sides.",
    74: "Buteo lineatus (Red-shouldered Hawk): Brown with reddish shoulders and barred underparts, often heard before seen.",
    75: "Zonotrichia leucophrys (White-crowned Sparrow): Black and white striped head with grayish underparts and pink bill.",
    76: "Saltator coerulescens (Grayish Saltator): Grayish body with a white throat, black bill, and a soft warbling song.",
    77: "Tyrannus vociferans (Cassin's Kingbird): Gray head, pale gray chest, and bright yellow belly with a black tail.",
    78: "Junco hyemalis (Dark-eyed Junco): Slate-gray head and chest with white belly, known for its distinctive white outer tail feathers.",
    79: "Pitangus sulphuratus (Great Kiskadee): Bright yellow belly, brown back, black and white striped head, and bold calls.",
    80: "Sylvia communis (Common Whitethroat): Small, grayish with a white throat and brownish back, often flicking tail up.",
    81: "Melanerpes erythrocephalus (Red-headed Woodpecker): Bright red head, black back, and white underparts with a square tail.",
    82: "Rhipidura leucophrys (Willie Wagtail): Small with black upperparts, white belly, and long tail, often seen wagging.",
    83: "Quiscalus major (Boat-tailed Grackle): Iridescent black feathers, long keel-shaped tail, and prominent white eyes (in males).",
    84: "Sphyrapicus varius (Yellow-bellied Sapsucker): Black and white with a red crown, males have a red throat and pale yellow belly.",
    85: "Sitta europaea (Eurasian Nuthatch): Grayish-blue back, white face, and underparts, often seen climbing tree trunks headfirst.",
    86: "Cyanocorax yncas (Green Jay): Bright green with a blue face and black bib, known for loud calls and social behavior.",
    87: "Corvus brachyrhynchos (American Crow): Large, all-black bird with a heavy bill and a fan-shaped tail.",
    88: "Sitta canadensis (Red-breasted Nuthatch): Small, with blue-gray back, reddish underparts, and a distinctive black eye stripe.",
    89: "Cyanistes caeruleus (Eurasian Blue Tit): Small with a blue cap, yellow belly, and black eye stripe with white cheeks.",
    90: "Colaptes auratus (Northern Flicker): Brown with black spots, a red nape, and a bright white rump visible in flight.",
    91: "Baeolophus inornatus (Oak Titmouse): Small, plain gray bird with a short crest, and a curious, active behavior.",
    92: "Erithacus rubecula (European Robin): Small with a distinctive orange-red chest and face, brown back, and white belly.",
    93: "Poecile carolinensis (Carolina Chickadee): Small with a black cap, bib, and white cheeks, found in wooded areas.",
    94: "Passer domesticus (House Sparrow): Chunky with brown and gray plumage, males have a black bib and white cheeks.",
    95: "Sphecotheres vieilloti (Australasian Figbird): Bright green (males) or brown (females) with a red or bare face patch.",
    96: "Mniotilta varia (Black-and-white Warbler): Black and white streaked plumage, often seen climbing tree trunks like a nuthatch.",
    97: "Dryocopus pileatus (Pileated Woodpecker): Large, mostly black with a bright red crest and white stripes on the face and neck.",
    98: "Prunella modularis (Dunnock): Brown with streaked back and pale gray underparts, often seen skulking low in vegetation.",
    99: "Megarynchus pitangua (Boat-billed Flycatcher): Bright yellow belly, black head with white eyebrows, and a massive, flattened bill.",
    100: "Melospiza georgiana (Swamp Sparrow): Brown, streaked with grayish face, and a rusty crown. Typically found in marshy areas.",
    101: "Agelaius phoeniceus (Red-winged Blackbird): Black plumage with bright red and yellow shoulder patches in males.",
    102: "Lanius cristatus (Brown Shrike): Brownish back, pale underparts, and black mask across the eyes.",
    103: "Catharus ustulatus (Swainson's Thrush): Brown upperparts, pale underparts with buffy eye ring. Known for its melodious song.",
    104: "Ammodramus savannarum (Grasshopper Sparrow): Small, with yellowish face, short tail, and streaked brown upperparts.",
    105: "Thryothorus ludovicianus (Carolina Wren): Rusty-brown body with a white eyebrow and a rounded tail often held up.",
    106: "Troglodytes aedon (House Wren): Small, brown, and streaked with a short tail often cocked upwards.",
    107: "Setophaga caerulescens (Black-throated Blue Warbler): Blue upperparts in males, with a black throat and face mask.",
    108: "Neochmia temporalis (Red-browed Finch): Small, olive-gray with a red eyebrow, rump, and bill.",
    109: "Setophaga citrina (Hooded Warbler): Black hood and bright yellow face in males, greenish back and yellow underparts.",
    110: "Myiodynastes luteiventris (Sulphur-bellied Flycatcher): Yellowish belly, streaked brown back, and bold black mask.",
    111: "Setophaga discolor (Prairie Warbler): Yellow with streaked sides, olive back, and rufous streaks on the back.",
    112: "Pipilo erythrophthalmus (Eastern Towhee): Black upperparts in males, rufous sides, white belly, and red eye.",
    113: "Pica nuttalli (Yellow-billed Magpie): Black and white with a yellow bill and greenish iridescent wings.",
    114: "Toxostoma longirostre (Long-billed Thrasher): Brown with a long, down-curved bill and streaked breast.",
    115: "Ardea melanocephala (Black-headed Heron): Gray with a black head, slender neck, and long legs. Often found in wetlands.",
    116: "Microcarbo africanus (Long-tailed Cormorant): Black with greenish sheen, slender neck, and long tail.",
    117: "Charadrius wilsonia (Wilson's Plover): Brown back, white underparts, and a large black bill.",
    118: "Spizella breweri (Brewer's Sparrow): Small and brown with a pale, streaked back and a finely marked head.",
    119: "Patagioenas flavirostris (Red-billed Pigeon): Grayish with a red bill and pale pinkish belly.",
    120: "Chlidonias hybrida (Whiskered Tern): Gray body with a black cap and white whisker-like streak on the cheek.",
    121: "Sterna paradisaea (Arctic Tern): Sleek, with a red bill, white body, and long tail streamers.",
    122: "Calidris virgata (Surfbird): Sturdy shorebird with grayish back, white underparts, and yellowish legs.",
    123: "Acridotheres cristatellus (Crested Myna): Blackish body with a short crest, white wing patches, and yellow eye patch.",
    124: "Tigrisoma lineatum (Rufescent Tiger Heron): Reddish-brown with streaked neck and back. Often seen in dense forests.",
    125: "Ciconia nigra (Black Stork): Large, with black body, red beak, and white underparts.",
    126: "Oriolus chinensis (Black-naped Oriole): Yellow with black nape and eye stripe, and a slender, red bill.",
    127: "Chloroceryle amazona (Amazon Kingfisher): Green body, white belly, and large, pointed bill.",
    128: "Sylvia curruca (Lesser Whitethroat): Grayish-brown with a whitish throat and subtle, pale eye ring.",
    129: "Pernis ptilorhynchus (Oriental Honey Buzzard): Brown with broad wings and a small head, often seen soaring.",
    130: "Charadrius alexandrinus (Kentish Plover): Brown above, white below, with a black forehead bar and slender build.",
    131: "Certhia brachydactyla (Short-toed Treecreeper): Small, brown with a slender downcurved bill, and white underparts.",
    132: "Myiodynastes maculatus (Streaked Flycatcher): Brown streaked back and wings, yellow belly, and a stout bill.",
    133: "Larus pacificus (Pacific Gull): Large gull with a stout yellow bill with red spot, black wings, and white body.",
    134: "Treron vernans (Pink-necked Green Pigeon): Green body, with pinkish neck (males) and yellowish belly.",
    135: "Urocissa erythroryncha (Red-billed Blue Magpie): Blue body, long tail, and red bill with black head.",
    136: "Microcarbo pygmeus (Pygmy Cormorant): Small, black with glossy greenish sheen, and a long, slender neck.",
    137: "Elseyornis melanops (Black-fronted Dotterel): White face with a black mask, slender bill, and brownish back.",
    138: "Mycteria leucocephala (Painted Stork): White body with pink tertial feathers and a yellow bill.",
    139: "Gerygone igata (Grey Warbler): Small, brownish-gray with pale eye ring, often seen fluttering in shrubs.",
    140: "Fulica cristata (Red-knobbed Coot): Black with red knobs on forehead, white bill, and lobed toes.",
    141: "Turdus leucomelas (Pale-breasted Thrush): Brown with pale, buffy underparts and a yellowish bill.",
    142: "Brotogeris jugularis (Orange-chinned Parakeet): Small, green, with a faint orange spot on the chin.",
    143: "Pycnonotus aurigaster (Sooty-headed Bulbul): Grayish body with black head and red undertail.",
    144: "Lagonosticta senegala (Red-billed Firefinch): Small, red face and bill, gray body with subtle white spots.",
    145: "Poecile hudsonicus (Boreal Chickadee): Brown cap, gray body, and white cheeks with a slightly darker bib.",
    146: "Ploceus velatus (Southern Masked Weaver): Yellow body, black mask, and distinctive nest-building behavior.",
    147: "Ploceus cucullatus (Village Weaver): Bright yellow with a black head and red eyes, often seen weaving nests.",
    148: "Plocepasser mahali (White-browed Sparrow-Weaver): Brown with a white eyebrow, pale underparts, and streaked back.",
    149: "Calidris ruficollis (Red-necked Stint): Small with rusty neck during breeding season, pale belly, and short bill.",
    150: "Tringa stagnatilis (Marsh Sandpiper): Slender, with long legs, grayish upperparts, and a straight bill.",
    151: "Empidonax virescens (Acadian Flycatcher): Olive-green with pale eye ring, yellowish belly, and short tail.",
    152: "Polemaetus bellicosus (Martial Eagle): Large, dark brown with spotted belly, powerful build, and broad wings.",
    153: "Lamprotornis superbus (Superb Starling): Iridescent blue-green with chestnut belly and white breast band.",
    154: "Jynx torquilla (Eurasian Wryneck): Brown with intricate patterns, long tail, and twisting head movements.",
    155: "Buteo rufinus (Long-legged Buzzard): Large, pale brown with a long tail and broad wings, often seen soaring.",
    156: "Cecropis daurica (Red-rumped Swallow): Reddish rump, glossy blue back, and white belly with faint streaks.",
    157: "Anser fabalis (Bean Goose): Brown with orange legs, and a black bill with an orange band.",
    158: "Coscoroba coscoroba (Coscoroba Swan): White body with black wingtips and red bill and legs.",
    159: "Anthus petrosus (Rock Pipit): Brownish-gray with streaked back and underparts, often seen on rocky shores.",
    160: "Torgos tracheliotos (Lappet-faced Vulture): Large with pinkish face, dark body, and a heavy, hooked bill.",
    161: "Lamprotornis nitens (Cape Starling): Iridescent blue-green with orange eyes and social behavior.",
    162: "Habia fuscicauda (Red-throated Ant Tanager): Brownish body with a red throat and head (in males).",
    163: "Calonectris diomedea (Scopoli's Shearwater): Brownish-gray with white belly and a slender, hooked bill.",
    164: "Vanellus senegallus (African Wattled Lapwing): Brown with yellow wattles, white forehead, and long, yellow legs.",
    165: "Polioptila californica (California Gnatcatcher): Small, grayish with a black tail and a white eye ring.",
    166: "Cinnyris asiaticus (Purple Sunbird): Glossy purplish-black with long, curved bill and yellowish underparts (in males).",
    167: "Saltator aurantiirostris (Golden-billed Saltator): Olive-gray with yellow bill, black around the eye, and white throat.",
    168: "Burhinus vermiculatus (Water Thick-knee): Brown with yellow eyes and long legs, found near water.",
    169: "Charadrius collaris (Collared Plover): Brown above, white below, with black collar and white eyebrow.",
    170: "Cyanocorax beecheii (Beechey’s Jay): Blue with black head, crest, and robust bill. Social and noisy.",
    171: "Myiozetetes cayanensis (Rusty-margined Flycatcher): Yellow belly, brownish upperparts, and a black mask.",
    172: "Dicrurus leucophaeus (Ashy Drongo): Gray with a long, forked tail and strong territorial calls.",
    173: "Phoeniculus purpureus (Green Wood Hoopoe): Glossy green-black with long, red, down-curved bill.",
    174: "Colaptes chrysoides (Gilded Flicker): Pale brown with black spots, red malar stripe (males), and yellow wing linings.",
    175: "Lophaetus occipitalis (Long-crested Eagle): Black with long crest feathers and white wing patches.",
    176: "Ixobrychus minutus (Little Bittern): Small, with yellowish-brown and black wings, long neck, and stealthy habits.",
    177: "Ardea goliath (Goliath Heron): Tall, with grayish plumage, rufous neck, and large bill.",
    178: "Stercorarius longicaudus (Long-tailed Jaeger): Slender, pale, with long central tail feathers and dark cap.",
    179: "Aethopyga siparaja (Crimson Sunbird): Bright crimson head and back (males) with greenish wings.",
    180: "Prinia subflava (Tawny-flanked Prinia): Tawny flanks, grayish-brown back, and a long, often flicking tail.",
    181: "Fluvicola nengeta (Masked Water-Tyrant): Black and white with a masked face pattern and a thin, pointed bill.",
    182: "Lophoceros alboterminatus (Crowned Hornbill): Black with red bill tipped in ivory, white throat patch, and crest.",
    183: "Vireo atricapilla (Black-capped Vireo): Olive-green with black head, white eye ring, and pale yellow underparts.",
    184: "Halcyon senegalensis (Woodland Kingfisher): Blue and white with bright red bill and black wings.",
    185: "Stelgidopteryx ruficollis (Southern Rough-winged Swallow): Brownish with buffy underparts and a squared tail.",
    186: "Ortalis cinereiceps (Gray-headed Chachalaca): Grayish head, brown body, and long, greenish tail.",
    187: "Forpus conspicillatus (Spectacled Parrotlet): Small, green with blue wings and white eye ring.",
    188: "Trogon rufus (Black-throated Trogon): Green back, black throat, and yellow belly. Short tail with white markings.",
    189: "Anthracothorax nigricollis (Black-throated Mango): Dark with green throat, black belly, and iridescent back.",
    190: "Rhipidura javanica (Pied Fantail): Black with white underparts, long tail, and distinct white eyebrow.",
    191: "Dendrocopos syriacus (Syrian Woodpecker): Black and white with a red crown in juveniles, white face markings.",
    192: "Eucometis penicillata (Gray-headed Tanager): Gray head, olive-brown body, and a pale yellow belly.",
    193: "Poecile sclateri (Mexican Chickadee): Black cap and bib, pale cheeks, and olive-brown body.",
    194: "Emberiza rustica (Rustic Bunting): Black and white with a chestnut breast and pale underparts.",
    195: "Colaptes punctigula (Spot-breasted Woodpecker): Yellow with black spots on breast and wings, and red cap (males).",
    196: "Ocreatus underwoodii (Booted Racket-tail): Green body with racket-tipped tail feathers, white leg tufts.",
    197: "Prionops plumatus (White Helmetshrike): Black with white helmet-like feathers, yellow wattles around eyes.",
    198: "Antrostomus vociferus (Eastern Whip-poor-will): Brown with intricate patterning and distinctive calls.",
    199: "Spheniscus magellanicus (Magellanic Penguin): Black above, white below, with two black bands across the chest."
}



