# Self-Supervised Learning for Building Damage Assessment

This repository contains code for the paper "Self-Supervised Learning for Building Damage Assessment from Large-scale xBD Satellite Imagery Benchmark Datasets".  [Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-12423-5_29)


The code is based on [DINO](https://github.com/facebookresearch/dino). The framework pretrains convolutional neural networks on unlabeled remote sensing imagery before fine-tuning on limited labels for building damage classification.

## Method

- Models like ResNet and ViT are pretrained on satellite images using self-supervision 
- The pretrained models are fine-tuned on 1% or 20% labeled data for building damage assessment
- Self-supervision enables effective learning from limited labels by using abundant unlabeled remote sensing data

## Usage

The pretrained models need to be configured with proper `hidden_size` and `sample_ind` hyperparameters based on the architecture:

1. To pretrain a resnet18 model,
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_path /path/to/image \
--batch_size_per_gpu 32 \
--num_worker 4 \
--arch resnet18 \
--hidden_size 512 256 128 64 \
--sample_ind 4 3 2 1 0
```
2. To pretrain a vit_small model,
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_path /path/to/image \
--batch_size_per_gpu 32\
 --num_worker 4 \
 --arch vit_small \
 --hidden_size 384 384 384 \
 --sample_ind 11 8 5 2 
```
3. To pretrain a vit_base model,
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_path /path/to/image \
--batch_size_per_gpu 32\
 --num_worker 4 \
 --arch vit_small \
 --hidden_size 768 768 768 \
 --sample_ind 11 8 5 2 
```
See the paper for full training details.

## Results
| Labeled Data | Method | Localization | Damage | No Damage | Minor | Major | Destroyed |
|-|-|-|-|-|-|-|-|
| 1% | ImageNet | 0.461 | 0.321 | 0.387 | 0.136 | 0.234 | NaN |  
|  | DINO | 0.522 | 0.366 | 0.480 | 0.157 | 0.439 | NaN |
|  | MoCo v3 | 0.550 | 0.379 | 0.425 | 0.124 | 0.337 | NaN |
|  | Ours | 0.539 | 0.390 | 0.486 | 0.261 | 0.345 | NaN |
| 20% | ImageNet | 0.661 | 0.587 | 0.604 | 0.278 | 0.471 | 0.456 |
|  | DINO | 0.714 | 0.601 | 0.667 | 0.229 | 0.384 | 0.447 |  
|  | MoCo v3 | 0.650 | 0.639 | 0.562 | 0.230 | 0.392 | 0.400 |
|  | Ours | 0.678 | 0.636 | 0.646 | 0.314 | 0.480 | 0.380 |

Table: F1 scores on building damage assessment using limited labeled data. Our method achieves competitive performance.

## Citation

If you find our work useful, please cite:
```
@inproceedings{Xia2022SelfSupervisedLF,
  title={Self-Supervised Learning for Building Damage Assessment from Large-scale xBD Satellite Imagery Benchmark Datasets},
  author={Zaishuo Xia and Zelin Li and Yanbing Bai and Jinze Yu and Bruno Adriano},
  booktitle={International Conference on Database and Expert Systems Applications},
  year={2022}
}
```
