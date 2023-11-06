# Enhanced Semantic Similarity Learning Framework for Image-Text Matching

<img src="docs/assets/img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Official PyTorch implementation of the paper [Enhanced Semantic Similarity Learning Framework for Image-Text Matching](https://www.researchgate.net/publication/373318149_Enhanced_Semantic_Similarity_Learning_Framework_for_Image-Text_Matching).

Please use the following bib entry to cite this paper if you are using any resources from the repo.

```
@article{zhang2023enhanced,
  title={Enhanced Semantic Similarity Learning Framework for Image-Text Matching},
  author={Zhang, Kun and Hu, Bo and Zhang, Huatian and Li, Zhe and Mao, Zhendong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```


We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty/blob/master/README.md) to build up our codebase. 


## Introduction

<img src="https://github.com/CrossmodalGroup/ESL/blob/main/overview.png" width="100%">


### Image-text Matching Results

The following tables show partial results of image-to-text retrieval on COCO and Flickr30K datasets. In these experiments, we use BERT-base as the text encoder for our methods. This branch provides our code and pre-trained models for **using BERT as the text backbone**, please check out to [**the ```CLIP-based``` branch**](https://github.com/woodfrog/vse_infty/tree/bigru) for the code and pre-trained models.

#### Results of 5-fold evaluation on COCO 1K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|Rsum|Link|
|---|:---:|:---:|---|---|---|---|---|---|---|---|
|ESL-H | BUTD region |BERT-base|**82.5**|**97.4**|**99.0**|**66.2**|**91.9**|**96.7**|**533.5**|[Here](https://drive.google.com/file/d/1NgTLNFGhEt14YgLb3gCkWfBp1gBxvl9w/view?usp=sharing)|
|ESL-A | BUTD region |BERT-base|**82.2**|**96.9**|**98.9**|**66.5**|**92.1**|**96.7**|**533.4**|[Here](https://drive.google.com/file/d/17jaJm2DSJbF5IuUij9s3c2fupcy4CW8T/view?usp=sharing)|


#### Results of 5-fold evaluation on COCO 5K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|Rsum|Link|
|---|:---:|:---:|---|---|---|---|---|---|---|---|
|ESL-H | BUTD region |BERT-base|**63.6**|**87.4**|**93.5**|**44.2**|**74.1**|**84.0**|**446.9**|[Here](https://drive.google.com/file/d/1NgTLNFGhEt14YgLb3gCkWfBp1gBxvl9w/view?usp=sharing)|
|ESL-A | BUTD region |BERT-base|**63.0**|**87.6**|**93.3**|**44.5**|**74.4**|**84.1**|**447.0**|[Here](https://drive.google.com/file/d/17jaJm2DSJbF5IuUij9s3c2fupcy4CW8T/view?usp=sharing)|


#### Results on Flickr30K Test Split

| |Visual Backbone|Text Backbone|R1|R5|R10|R1|R5|R10|Rsum|Link|
|---|:---:|:---:|---|---|---|---|---|---|---|---|
|ESL-H | BUTD region |BERT-base|**83.5**|**96.3**|**98.4**|**65.1**|**87.6**|**92.7**|**523.7**|[Here](https://drive.google.com/file/d/17FnwyH8aSOwvUuZco0lQ5eY0TM_4LXxv/view?usp=sharing)|
|ESL-A | BUTD region |BERT-base|**84.3**|**96.3**|**98.0**|**64.1**|**87.4**|**92.2**|**522.4**|[Here](https://drive.google.com/file/d/1ZoPW8azNkBWVq1jaQxHfI_XpINzvmv1n/view?usp=sharing)|





## Preparation

### Environment

We trained and evaluated our models with the following key dependencies:

- Python 3.7.3 

- Pytorch 1.2.0

- Transformers 2.1.0


Run ```pip install -r requirements.txt ``` to install the exactly same dependencies as our experiments. However, we also verified that using the latest Pytorch 1.8.0 and Transformers 4.4.2 can also produce similar results.  

### Data

We organize all data used in the experiments in the following manner:

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── images   # raw coco images
│   │      ├── train2014
│   │      └── val2014
│   │
│   ├── cxc_annots # annotations for evaluating COCO-trained models on the CxC benchmark
│   │
│   └── id_mapping.json  # mapping from coco-id to image's file name
│   
│
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── ......
│   │
│   ├── flickr30k-images   # raw coco images
│   │      ├── xxx.jpg
│   │      └── ...
│   └── id_mapping.json  # mapping from f30k index to image's file name
│   
├── weights
│      └── original_updown_backbone.pth # the BUTD CNN weights
│
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)
```

The download links for original COCO/F30K images, precomputed BUTD features, and corresponding vocabularies are from the offical repo of [SCAN](https://github.com/kuanghuei/SCAN#download-data). The ```precomp``` folders contain pre-computed BUTD region features, ```data/coco/images``` contains raw MS-COCO images, and ```data/f30k/flickr30k-images``` contains raw Flickr30K images. 

The ```id_mapping.json``` files are the mapping from image index (ie, the COCO id for COCO images) to corresponding filenames, we generated these mappings to eliminate the need of the ```pycocotools``` package. 

```weights/original_updowmn_backbone.pth``` is the pre-trained ResNet-101 weights from [Bottom-up Attention Model](https://github.com/peteanderson80/bottom-up-attention), we converted the original Caffe weights into Pytorch. Please download it from [this link](https://drive.google.com/file/d/1gNdV1Qx_7yYzkhHrzqbP-bbNkdrKw_w1/view?usp=sharing).


The ```data/coco/cxc_annots``` directory contains the necessary data files for running the [Criscrossed Caption (CxC) evaluation](https://github.com/google-research-datasets/Crisscrossed-Captions). Since there is no official evaluation protocol in the CxC repo, we processed their raw data files and generated these data files to implement our own evaluation.  We have verified our implementation by aligning the evaluation results of [the official VSRN model](https://github.com/KunpengLi1994/VSRN) with the ones reported by the [CxC paper](https://arxiv.org/abs/2004.15020) Please download the data files at [this link](https://drive.google.com/drive/folders/1Ikwge0usPrOpN6aoQxsgYQM6-gEuG4SJ?usp=sharing).

Please download all necessary data files and organize them in the above manner, the path to the ```data``` directory will be the argument to the training script as shown below.

## Training

Assuming the data root is ```/tmp/data```, we provide example training scripts for:

1. Grid feature with BUTD CNN for the image feature, BERT-base for the text feature. See ```train_grid.sh```

2. BUTD Region feature for the image feature, BERT-base for the text feature. See ```train_region.sh```


To use other CNN initializations for the grid image feature, change the ```--backbone_source``` argument to different values: 

- (1). the default ```detector``` is to use the [BUTD ResNet-101](https://github.com/peteanderson80/bottom-up-attention), we have adapted the original Caffe weights into Pytorch and provided the download link above; 
- (2). ```wsl```  is to use the backbones from [large-scale weakly supervised learning](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/); 
- (3). ```imagenet_res152``` is to use the ResNet-152 pre-trained on ImageNet. 



## Evaluation

Run ```eval.py``` to evaluate specified models on either COCO and Flickr30K. For evaluting pre-trained models on COCO, use the following command (assuming there are 4 GPUs, and the local data path is /tmp/data):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset coco --data_path /tmp/data/coco
```

For evaluting pre-trained models on Flickr-30K, use the command: 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset f30k --data_path /tmp/data/f30k
```

For evaluating pre-trained COCO models on the CxC dataset, use the command:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval.py --dataset coco --data_path /tmp/data/coco --evaluate_cxc
```


For evaluating two-model ensemble, first run single-model evaluation commands above with the argument ```--save_results```, and then use ```eval_ensemble.py``` to get the results (need to manually specify the paths to the saved results). 



