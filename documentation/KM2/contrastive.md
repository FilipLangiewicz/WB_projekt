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
It is not clear how to augument images in the multispectral setting. Some simple methods of augumentation as image clipping or adding noise may not be as effective as in RGB settings.


## Experiments
All experiments were carried on similar settings - the used dataset was EuroSAT with 13 bands. Models were trained on 200 epochs and with output embedding dimensions of size 512.

### Contrastive tile2vec
Firstly, we tried to leverage the already written tile2vec code to be used with different objective. We simply modified the triplet loss to analogous contrastive loss that compared only two images - the anchor and the neighbour. Since the proposed model described in [the original paper](https://arxiv.org/abs/1805.02855) samples both the anchor and neighbour tile from the same image, there was little to be done to swap the triplet loss to the simpler, contrastive one.

The tile2vec model, on the preprocessing step does already some simple data augumentation, such as image flipping and rotation. The tile size used in the sampling process equaled to 50x50 pixels, therefore the anchor and neighbour could be treated as views of the same image.
### SimCLR

Secondly, we used an approach proposed in [SimCLR](https://arxiv.org/pdf/2002.05709) article witch describes a simple contrastive learning framework. In our setting we used the ResNet18 architecture with default hyperparameters and no projection head. Two views of the input image were augumented using two different stochastic function calls, which consisted of:
1. random croping
2. resizing the image back to the original size
3. random color distortion
4. random gaussian blur
which are described in detail in the SimCLR article.


### Divmaker
Using the ideas and code from the [divmaker](https://arxiv.org/pdf/2302.05757) article, we created an unsupervised framework, leveraging the SimCLR model and were generating the augumented views of the images using a additional net - *divmaker* trained to create as diverse image views as possible along with the embedding model.

## Results

### Contrastive tile2vec
Without suprise the simple modification of tile2vec model perfomed very poorly. Even with large regularization and modification of the loss - cosine similarity instead of L2 distance of embeddings, the model perfomance was much worse, even comparing to the Tile2vec model trained on triplet of images.

The reason of this behaviour lies probably in insufficient data augumentation not enough diversity between the following epochs. Following the original tile2vec implementation, tiles were prepared before the training and as a result model could seen less variation within the dataset.

### SimCLR
Model performed suprisingly well. Even though the used augumentations were not designed for the multispectral images, but RGB samples, the embeddings created by the model contained informations of better quality - the random forest trained on the embeddings outperformed the one trained on tile2vec model with triplet loss



### Divmaker
The novel augumentation of data, aimed at its diversity has resulted in even better results compared to SimCLR. The predictive power of Random Forest trained at the embeddings of this model was the largest.
Even though with less epochs it achieved performance for RF around 92% for 512 dimensions and 96% for 512 which is similar to the SOTA viewmaker of 2021.


## Summary
The contrastive learning in our setting did very well. Different augumentations of data resulted in much better performance of produced embeddings, which at the end contain enough information for the simpler classifier to achieve very big accuracy.


#TODO - add some more details




