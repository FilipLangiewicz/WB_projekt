# Contrastive unsupervised learning

As a part of second milestone, we decided to explore some modifications of the tile2vec model trained on EuroSAT dataset. One of the possible methods was replacing the triplet loss with some different objective. 

## Problems with triplet loss

As mentioned in [Self-Supervised Learning for Scene Classification in Remote Sensing: Current State of the Art and Perspectives](https://hal.science/hal-03934160/document) - a review of methods used in multispectral imaginery:
Applying this method requires either having the coordinate location of each image from the dataset or having very-high-resolution images that can be divided into smaller patches.

Unfortunately our dataset contains very small (64x64 pixels) patches, so in order to make tile2vec work better (in unsupervised paradigm), we would need to leverage the information about the images geolocalizations, or simply try different approach to unsupervised learning.

## Simple contrastive unsupervised learning
In this subtask, we switched to a simpler objective - an usual contrastive loss that compares two images that were augumented in different ways.

### Challenges

#### Data augumentation
It is not clear how to augument images in the multispectral setting.


