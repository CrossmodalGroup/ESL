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
<div align=center><img src="https://github.com/CrossmodalGroup/ESL/blob/main/motivation.png" width="50%" ></div>
  
Squares denote local dimension elements in a feature. Circles denote the measure-unit, i.e., the minimal basic component used to examine semantic similarity. Compared with (a) existing methods typically default to a static mechanism that only examines the single-dimensional cross-modal correspondence, (b) our key idea is to dynamically capture and learn multi-dimensional enhanced correspondence.  That is, the number of dimensions constituting the measure-units is changed from existing only one to hierarchical multi-levels, enabling their examining information granularity to be enriched and enhanced to promote a more comprehensive semantic similarity learning.

## Introduction
<img src="https://github.com/CrossmodalGroup/ESL/blob/main/overview.png" width="100%">
In this paper, different from the single-dimensional correspondence with limited semantic expressive capability, we propose a novel enhanced semantic similarity learning (ESL), which generalizes both measure-units and their correspondences into a dynamic learnable framework to examine the multi-dimensional enhanced correspondence between visual and textual features. Specifically, we first devise the intra-modal multi-dimensional aggregators with iterative enhancing mechanism, which dynamically captures new measure-units integrated by hierarchical multi-dimensions, producing diverse semantic combinatorial expressive capabilities to provide richer and discriminative information for similarity examination. Then, we devise the inter-modal enhanced correspondence learning with sparse contribution degrees, which comprehensively and efficiently determines the cross-modal semantic similarity. Extensive experiments verify its superiority in achieving state-of-the-art performance.

### Image-text Matching Results

The following tables show partial results of image-to-text retrieval on COCO and Flickr30K datasets. In these experiments, we use BERT-base as the text encoder for our methods. This branch provides our code and pre-trained models for **using BERT as the text backbone**. Some results are better than those reported in the paper. However, it should be noted that the ensemble results in the paper are not obtained by the best two checkpoints provided. It is lost due to not saving in time. You can train the model several times more and then combine any two to find the best ensemble performance. Please check out to [**the ```CLIP-based``` branch**](https://github.com/kkzhang95/ESL/blob/main/README.md) for the code and pre-trained models.

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
* The specific required environment can be found [here](https://github.com/CrossmodalGroup/ESL/blob/main/ESL.yaml) Using **conda env create -f ESL.yaml** to create the corresponding environments.

### Data

Download the dataset files. We use the image feature created by SCAN. The vocabulary required by GloVe has been placed in the 'vocab' folder of the project (for Flickr30K and MSCOCO).

You can download the dataset through Baidu Cloud. Download links are [Flickr30K]( https://pan.baidu.com/s/1Fr_bviuWLcrJ9MiiRn_H2Q) and [MSCOCO]( https://pan.baidu.com/s/1vp3gtQhT7GO0PQACBSnOrQ), the extraction code is: USTC. 

## Training

```bash
sh  train_region_f30k.sh
```

```bash
sh  train_region_coco.sh
```
For the dimensional selective mask, we design both heuristic and adaptive strategies.  You can use the flag in [vse.py](https://github.com/CrossmodalGroup/ESL/blob/main/lib/vse.py) (line 44) 
```bash
heuristic_strategy = False
```
to control which strategy is selected. True -> heuristic strategy, False -> adaptive strategy. 

## Evaluation

Test on Flickr30K
```bash
python test.py
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

```bash
python testall.py
```

To ensemble model, specify the model_path in test_stack.py, and run
```bash
python test_stack.py
```


