# BQNet: Boosting Query Points Network for 3D Referring Expression Segmentation

Framework:

<img src="docs\framework.png"/>

## Introduction
3D Referring Expression Segmentation (3D-RES) localizes and segments objects within 3D point clouds based on natural language descriptions. A critical challenge lies in the quality of query point features, as they directly interact with scene features to generate segmentation masks. Despite recent progress, existing methods suffer from three key limitations: inadequate progressive fusion of multimodal features, insufficient modeling of explicit spatial relations, and uneven learning difficulties among query points due to varying scene complexities. To overcome these limitations, we introduce Boosting Query points Network (BQNet). BQNet iteratively optimizes query point features using three core modules: the Progressive Multimodal Knowledge Learning (PMKL) module, which progressively fuses multimodal knowledge; the Dual Factor Attention Block (DFB) for explicitly modeling feature relationships and spatial context; and the Foreground-Background Hard Query point Learning (FBHQL) module, which optimizes challenging query points through adaptive difficulty perception mechanism. Comprehensive experiments demonstrate BQNet's superiority over state-of-the-art by 1.3 and 3.0 points in Acc@0.5 metrics on the 3D-RES and 3D-GRES tasks, respectively.

## Installation

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n BQNet python=3.8
  conda activate BQNet
  ```

- Clone this repository

- If you have not cmake
  ```
  #cmake
  wget https://github.com/Kitware/CMake/releases/download/v3.29.1/cmake-3.29.1.tar.gz
  tar -xzvf cmake-3.29.1.tar.gz
  cd cmake-3.29.1

  ./bootstrap --prefix=$HOME/local --parallel=8

  make -j8

  make install

  vim ~/.bashrc

  export LOCAL_PATH="$HOME/local/bin"
  export PATH="$LOCAL_PATH:$PATH"

  source ~/.bashrc

  cmake --version
  ```

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl # please check the versions in the website
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (download this repo).
  ```
  cd segmentator/csrc && mkdir build && cd build

  cmake .. \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
  -DCUDA_TOOLKIT_ROOT_DIR=$HOME/cuda-11.3

  make && make install # after install, please do not delete this folder (as we only create a symbolic link)
  cd ../../../


  cd csrc && mkdir build && cd build

  cmake .. \
  -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
  -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 

  make && make install # after install, please do not delete this folder (as we only create a symbolic link)
  ```

- Setup, Install BQNet and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd bqnet/lib/
  python setup.py develop
  cd ../../
  ```

- If you do not have permission for the following command "sudo apt-get install libsparsehash-dev"

    ```
    wget https://github.com/sparsehash/sparsehash/archive/refs/tags/sparsehash-2.0.4.tar.gz
    tar -xvzf sparsehash-2.0.4.tar.gz
    cd sparsehash-sparsehash-2.0.4

    ./configure --prefix=$HOME/.local 

    make        
    make install 

    echo 'export CPLUS_INCLUDE_PATH="$HOME/.local/include:$CPLUS_INCLUDE_PATH"' >> ~/.bashrc
    echo 'export LIBRARY_PATH="$HOME/.local/lib:$LIBRARY_PATH"' >> ~/.bashrc
    source ~/.bashrc

    ls ~/.local/include/sparsehash 
    ```

- Compile pointnet++
  ```
  cd pointnet2
  python setup.py install --user
  cd ..
  ```
- Install Pytorch3D. You can try 'pip install pytorch3d'. If it doesn't work, you can install it from source:
  ```
  git clone git@github.com:facebookresearch/pytorch3d.git
  cd pytorch3d && pip install -e .
  ```

- Install Spcay
  ```
  pip install spacy==3.7.5
  pip install pydantic==2.10.6
  ```
- Install [en_core_web_sm-3.7.1.tar.gz]()
  ```
  download it

  pip intall en_core_web_sm-3.7.1.tar.gz
  ```

## Data Preparation

## [ALL Data](https://pan.baidu.com/s/17QjG11xik8WsTS-u20202w?pwd=yayh)

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` folder as follows. You need to download the ['.aggregation.json', '.sens', '.txt', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean_2.labels.ply', '_vh_clean.aggregation.json'] files. (Please note ! ! ! This will take up nearly 1TB of storage space ! ! ! If you do not have enough hard drive capacity, you can skip downloading the '.sens' files and the 2D feature extraction step, and directly download the processed 2D features.)

```
IPDN
├── data
│   ├── scannetv2
│   │   ├── scans
```

Split and preprocess point cloud data (Note! If you have not downloaded the '.sens' files, please comment out the corresponding sections in the script before running it.)

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
MDIN
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── train
│   │   ├── val
│   │   ├── processed (if you process '.sen' file)
```

Obtain image features using CLIP and project them to point. (Please modify the output of the original CLIP's visual encoder to obtain 'tokens'. See line 79 in image2point_clip.py for more imformation.)

If you haven't processed the '.sen' files or find this step too time-consuming, you can download our preprocessed features and unzip them into the 'clip-feat' folder. Click [here](https://pan.baidu.com/s/11eeMZEeS0t7LZIeBDD_k1g?pwd=c8ct) to download.
```
cd ..
python image2point_clip.py
```

### ScanRefer dataset
Download [scanrefer](https://pan.baidu.com/s/1dayGht2PFqz8kx--Q2cZhA?pwd=n9s4) annotations following the instructions.

Put the downloaded `scanrefer` folder as follows.
```
IPDN
├── data
│   ├── scanrefer
│   │   ├── ScanRefer_train_new.json
│   │   ├── ScanRefer_val_new.json
```

### multi3drefer dataset
Downloading the [multi3drefer](https://pan.baidu.com/s/19nwg8175aSODXiqT7VAxrg?pwd=caf5) annotations. 
Put the downloaded `multi3drefer` folder as follows.
```
MDIN
├── data
│   ├── multi3drefer
│   │   ├── multi3drefer_train.json
│   │   ├── multi3drefer_val.json
```

## Pretrained Backbone

Download [SPFormer](https://pan.baidu.com/s/1it4h_M-9eIMXAsC2rSn7Dw?pwd=yy3x) pretrained model and move it to backbones.

```
mkdir backbones
mv ${Download_PATH}/sp_unet_backbone.pth backbones/
```

## Training
For 3D-GRES:
```
python tools/train.py configs/BQNet_gres.yaml --gpu_id 0
```
For 3D-RES:
```
python tools/train.py configs/BQNet_res.yaml --gpu_id 0
```

## Inference
For 3D-GRES:
```
python tools/train.py configs/BQNet_gres.yaml --checkpoint best_gres.pth --gpu_id 0
```
For 3D-RES:
```
python tools/train.py configs/BQNet_res.yaml --checkpoint best_res.pth --gpu_id 0
```

<!-- ## Citation

If you find this work useful in your research, please cite:

```

``` -->

## Models
Download pretrain models and move it to checkpoints.
|Benchmark | Task  | mIoU | Acc@0.25 | Acc@0.5 | Model |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Multi3DRes | 3D-GRES | 53.2 | 71.6 | 53.0 | [Model](https://pan.baidu.com/s/1u2sUramI2KyUvI-Io6FcKQ?pwd=tqc2) |
| ScanRefer   | 3D-RES | 50.9 | 61.4 | 56.2 | [Model](https://pan.baidu.com/s/1kRfm6s_aEmHU7cycf1uCTw?pwd=6hkb) |

## Ancknowledgement

Sincerely thanks for [MDIN](https://github.com/sosppxo/MDIN), [MaskClustering](https://github.com/PKU-EPIC/MaskClustering), [ReLA](https://github.com/henghuiding/ReLA), [M3DRef-CLIP](https://github.com/3dlg-hcvc/M3DRef-CLIP), [EDA](https://github.com/yanmin-wu/EDA), [SceneGraphParser](https://github.com/vacancy/SceneGraphParser), [SoftGroup](https://github.com/thangvubk/SoftGroup), [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet), [SPFormer](https://github.com/sunjiahao1999/SPFormer) and  [IPDN](https://github.com/80chen86/IPDN) repos. This repo is build upon them.

