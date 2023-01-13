# SSLBDA
A Self-Supervised Learning Framework for Remote Sensing.

[Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-12423-5_29)
Self-Supervised Learning for Building Damage Assessment from Large-scale xBD Satellite Imagery Benchmark Datasets

# Command Lines
The code is based on [DINO](https://github.com/facebookresearch/dino).

Note that the parameters hidden_size and sample_ind need to be adjusted with the network.
Here are a few examples.

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
