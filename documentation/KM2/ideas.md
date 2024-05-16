# Second Milestone

The goal of the second milestone was to make an interesting hypothesis about our model/data and then try to find answers and solutions for it. Therefore the first challenge was to come up with the ideas about the potential hypothesis.

## Main ideas
The general problem with our model was its weak performance on our dataset. Even though we tried to adjust the tile size and come up with different preprocessing methods, embedding created by the model had poor quality. Random forest trained on embeddings created by our model was performing worse than the ones trained on different dimension reduction methods.



### Selecting different, task specifics bands
Because of the poor model performance we wanted to check in which settings the model would perform better. [Domain knowledge](https://gisgeography.com/sentinel-2-bands-combinations/?fbclid=IwAR2LXmFCnhJAeyMbyBSZ8xpp9b2gYGDswTLFxjVpw2AjaRnyAEM8h4OATxA) suggests that different bands could be used to detect different categories.

We decided to pretrain model on different bands, to detect if model trained on bands related to vegetation will perform better when it comes to predicting specific classes: forests, etc.

### Noise at target regularization
[An article from 2021](https://www.researchgate.net/publication/356458961_Tile2vec_with_predicting_noise_for_land_cover_classification) suggests that L2 regularization implemented in the original article could be changed to Noise at Target objective, which slighlty imporoves the model performance.

We could implement this idea and evaluate the results on our dataset. The con of this solution is that the paper does not include any code.


### Representation from different classifier
The problem with the tile2vec embeddings might come not from our implementation but simply from the dataset or some different unknown source. The check this hypothesis we could pretrain a supervised classifier on our labeled Sentinel dataset.

Later, to compare the embeddings we could remove the last layer from the classifier and evaluate it as in first milestone. 

### Knowledge distillation
An easy task could be to simply take a large pretrained multispectral model (as [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)) and using knowledge distillation train a smaller one on our dataset, comparing the small model's performance with tile2vec.

There could be a problem with specific bands - large models are usually pretrained on different set of bands.

### Hard Negative mining
To enchance model we could modify the loss function details. As our dataset is already labeled, we could leverage the labels and pass this information to the model. 

For our case, this idea could be implemented as follows:
For all or some cases, the distant tile could be explicitly taken from the same class as the anchor tile.

maybe some interesting source: [False Negative Elimination](https://arxiv.org/pdf/2308.04380)


A different approach would be to leverage the labels in a triplet loss, such that neighbour tile could be taken from the same class as the anchor, meanwhile the distand could be taken from the different class.

### Self-supervised representation learning
The main goal of this method would be to train model with different objective. The objective would be to compare the model outputs from differently augumented image.

[this could be a source](https://medium.com/analytics-vidhya/self-supervised-representation-learning-in-computer-vision-part-2-8254aaee937c)




## Evaluation - How to evaluate the results

### Simple classifier on top
The basic idea proposed in the original article, was to train a classifier on top of the embeddings. 

### Clustering 
Alternatively we could train a clustering method (such as kmeans) and later on evaluate the performance on different embeddings.

Here we could use the silhouette, calinski harabasz along with homogenity and completness score.

### Visualisation

As during the first project we could create a tSNE visualisation which gives an insight how the representation is organised.

tSNE could be run on embeddings preprocessed by PCA

