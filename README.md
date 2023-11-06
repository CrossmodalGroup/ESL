# Enhanced Semantic Similarity Learning Framework for Image-Text Matching

<img src="https://github.com/CrossmodalGroup/ESL/blob/main/lib/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

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

## Motivation
<img src="https://github.com/CrossmodalGroup/ESL/blob/main/motivation.png" width="100%">
Squares denote local dimension elements in a feature. Circles denote the measure-unit, i.e., the minimal basic component used to examine semantic similarity. Compared with (a) existing methods typically default to a static mechanism that only examines the single-dimensional cross-modal correspondence, (b) our key idea is to dynamically capture and learn multi-dimensional enhanced correspondence.  That is, the number of dimensions constituting the measure-units is changed from existing only one to hierarchical multi-levels, enabling their examining information granularity to be enriched and enhanced to promote a more comprehensive semantic similarity learning.

## Introduction
<img src="https://github.com/CrossmodalGroup/ESL/blob/main/overview.png" width="100%">
In this paper, different from the single-dimensional correspondence with limited semantic expressive capability, we propose a novel enhanced semantic similarity learning (ESL), which generalizes both measure-units and their correspondences into a dynamic learnable framework to examine the multi-dimensional enhanced correspondence between visual and textual features. Specifically, we first devise the intra-modal multi-dimensional aggregators with iterative enhancing mechanism, which dynamically captures new measure-units integrated by hierarchical multi-dimensions, producing diverse semantic combinatorial expressive capabilities to provide richer and discriminative information for similarity examination. Then, we devise the inter-modal enhanced correspondence learning with sparse contribution degrees, which comprehensively and efficiently determines the cross-modal semantic similarity. Extensive experiments verify its superiority in achieving state-of-the-art performance.

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

We recommended the following dependencies.

* Python 3.6
* [PyTorch](http://pytorch.org/) 1.8.0
* [NumPy](http://www.numpy.org/) (>1.19.5)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* The specific required environment can be found [here](https://drive.google.com/file/d/1jLhd1GU6W3YrKeADM5g4qQxJoYt1lXx5/view?usp=sharing)

### Data

Download the dataset files. We use the image feature created by SCAN. The vocabulary required by GloVe has been placed in the 'vocab' folder of the project (for Flickr30K and MSCOCO).

You can download the dataset through Baidu Cloud. Download links are [Flickr30K]( https://pan.baidu.com/s/1Fr_bviuWLcrJ9MiiRn_H2Q) and [MSCOCO]( https://pan.baidu.com/s/1vp3gtQhT7GO0PQACBSnOrQ), the extraction code is: USTC. 

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



