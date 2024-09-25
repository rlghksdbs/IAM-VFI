# IAM-VFI : Interpolate Any Motion for Video Frame Interpolation with motion complexity map [ECCV 2024]

<div>
    <h4 align="center">
        <a href="https://rlghksdbs.github.io/iam-vfi_page" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
        </a>
    </h4>
</div>

## Installation
This repository is built in Docker

1. Clone our repository
    ```
    git clone https://github.com/rlghksdbs/IAM-VFI
    cd IAM-VFI
    ```

2. Create docker environment
The Docker environment used can be recreated using the ```Dockerfile```
    ```
    docker build --tag IAM-VFI .
    nvidia-docker run --name IAM-VFI -it --gpus all --ipc=host --pid=host -v /your/source_code/path/:/IAM-VFI/ --shm-size=64g IAM-VFI:latest
    ```

## Datasets Download
All the datasets for VFI used in the paper can be downloaded from the following locations:
#### Train Dataset : [Vimeo90K](http://toflow.csail.mit.edu/)
#### Test Datasets : [HD](https://github.com/baowenbo/MEMC-Net?tab=readme-ov-file), [SNU-FILM](https://myungsub.github.io/CAIN/), [UVG](https://ultravideo.fi/#testsequences), [Xiph2K&4K](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py)

## Prepare Data for training IAM-VFI
1. Generate 2X Scale vimeo dataset called VimeoX2 (upscale with [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch))

    The data should be placed in ```IAM-VFI/data```
The datasets are organized as:
    ```
    IAM-VFI
        └─data
            ├─01_vimeo_triplet
            │   └─sequences
            │       ├─00001
            │        ...
            ├─02_vimeo_triplet_x2
            │   └─sequences
            │       ├─00001
            │        ...
            └─all.txt
    ```
2. Classified Training Dataset based on TI

    #### 1) make patch data
    ```
    python img2patch.py
    ```
    #### 2) calculate TI
    ```
    python calculate_ti.py
    ```
    #### 3) data classification based on TI
    ```
    python split_data.py
    ```
3. after all process to prepare data the data should be placed in ```IAM-VFI/patch_data``` and the datasets are organized as:
    ```
    IAM-VFI
        └─patch_data
            ├─01_vimeo_triplet
            │   └─sequences
            │       ├─00001
            │           ├─0001
            │               ├─0_0
            │               └─0_1
            │            ...
            │           └─1000
            │        ...
            │       └─00078
            ├─02_vimeo_triplet_x2
            │   └─sequences
            │       ├─00001
            │           ├─0001
            │               ├─0_0
            │                ...
            │               └─2_4
            │            ...
            │           └─1000
            │        ...
            │       └─00078
            ├─train_text
            │   └─vimeo_x1x2_TI_15_30
            │       ├─all.txt            
            │       ├─easy.txt
            │       ├─medium.txt
            │       └─hard.txt
            ├─patch_ti.txt
            └─all_patch.txt
    ```
## Training
1. Train Optical Flow Network with classified Datasets based on TI
    
    - Train FlowNet with easy dataset
        ```
        pyhon train.py --datasets "ti_dataset" --ti_type "easy"
        ```    
    - Train FlowNet with medium dataset
        ```
        pyhon train.py --datasets "ti_dataset" --ti_type "medium"
        ```
    - Train FlowNet with hard dataset
        ```
        pyhon train.py --datasets "ti_dataset" --ti_type "hard"
        ```

2. Train Overall IAM-VFI Network
    ```
    python train.py --datasets "vimeo_triplet" --classifier_intra_v2 --easy_pkl_model_path ./ckpt/flownet_easy.pkl --medium_pkl_model_path ./ckpt/flownet_medium.pkl --hard_pkl_model_path ./ckpt/flownet_hard.pkl
    ```
## Inference
The code will be released soon
