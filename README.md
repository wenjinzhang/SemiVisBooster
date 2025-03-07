
## SemiVisBooster
Our algorithm implementation is at ```semilearn/algorithms/freematch/text_match.py```

### Prerequisites
Our Method is based on USB. USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```


### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.


### Training

Run experiment on Food101 dataset with 404 labeled images.

```sh
python train.py --c config/usb_cv/clipfreematch/textmatch_food101_404_0_labelname_mlp.yaml
```

More experiment can be found under folder ```config/usb_cv/clipfreematch/```

### Evaluation

After training, you can check the evaluation performance on training logs, or running evaluation script:

```
python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT
```

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

We thanks USB of creating our method:
- [USB](https://github.com/microsoft/Semi-supervised-learning/)
