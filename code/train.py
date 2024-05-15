from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet
from src.training import prep_triplets, train_triplet_epoch
from torch import optim

import os
import torch
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

print("Cuda device: ", cuda)


img_type = "landsat" #images are in float - this parameter specifies that there is a need for normalization of floats
tile_dir = '/storage/tile2vec/tiles'
bands = 13
augment = True
batch_size = 50
shuffle = True
num_workers = 16
n_triplets = 50000
z_dim = 512


dataloader = triplet_dataloader(img_type, tile_dir, bands=bands, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                n_triplets=n_triplets, pairs_only=True)
print('Dataloader set up complete.')



in_channels = bands


TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
TileNet.train()
if cuda: 
    TileNet.cuda()
print('TileNet set up complete.')


lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))



epochs = 50
margin = 10
l2 = 0.01
print_every = 1000
save_models = True

model_dir = '/storage/tile2vec/models'
if not os.path.exists(model_dir): 
    os.makedirs(model_dir)
    
results_fn = "/storage/tile2vec/results_fn"


t0 = time()
with open(results_fn, 'w') as file:

    print('Begin training.................')
    for epoch in range(0, epochs):
        (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
            TileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
            print_every=print_every, t0=t0)
        

# Save model after last epoch
if save_models:
    print("saving model")
    model_fn = os.path.join(model_dir, 'TileNet_default_clipping.ckpt')
    torch.save(TileNet.state_dict(), model_fn)