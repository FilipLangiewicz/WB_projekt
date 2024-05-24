# Idea

Training on triplets sometimes finds hard to learn difficult distant tiles. In our sampling images were taken randomly from given set. It was possible that anchor, neighbour and distant were all the same class. So in order to make model learn better harder triplets we redesinged function responsible for samling so that anchor and neighbour have diffrent class from the distant tile.

# Results

We created a new model with standard hyperparamters called "Tilenet_Distant_Diff.cptg". Results of the model compered to previous one are shown in table below.

| Embedding model   | Random Forest |
|-------------------|---------------|
| our tile2vec      | 58.88±0.73%   | 
| hard mining       | 65.92±0.71%   |

As we see, changing sampling method increased accuracy by 7pp.