
import sys
sys.path.append("..")

from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet
from src.training import prep_triplets, train_triplet_epoch
from torch import optim

import matplotlib.pyplot as plt
import pickle 

import os
import torch
from time import time

from pathlib import Path

# script to train the tile2vec model


# values to change during training
model_name = 'TileNet_no_triplets_reg.ckpt'
img_type = "landsat" # images are in float - this parameter specifies that there is a need for normalization of floats
tile_dir = Path('/storage/tile2vec/tiles') # directory where are the triplets stored
bands = 13
augment = True
batch_size = 50
shuffle = True
num_workers = 16
n_triplets = 50000
z_dim = 512


# initialize GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

print("Cuda device: ", cuda)


# dataloader in shor loades data for the model
dataloader = triplet_dataloader(img_type, tile_dir, bands=bands, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                n_triplets=n_triplets, pairs_only=True)
print('Dataloader set up complete.')


torch.autograd.set_detect_anomaly(True)
# function to initialize model structure
TileNet = make_tilenet(in_channels=bands, z_dim=z_dim, triplet=False, cosine=True)
TileNet.train()
if cuda: 
    TileNet.cuda()
print('TileNet set up complete.')

# set up the learning rate and select the optimizer
lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))


# training-level parameters 
epochs = 200
margin = 0.5
l2 = 1
print_every = 1000 # how often model will produce the information about the loss
save_models = True

# create model directory
model_dir = Path('/storage/tile2vec/models')
if not os.path.exists(model_dir): 
    os.makedirs(model_dir)
    

results_path = Path("/storage/tile2vec/results")

avg_losses = []
avg_l_ns = []
avg_l_ds = []

t0 = time()
print('Begin training.................')
for epoch in range(0, epochs):
        (avg_loss, avg_l_n, avg_l_d) = train_triplet_epoch(
            TileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
            print_every=print_every, t0=t0)
        
        avg_losses.append(avg_loss)
        avg_l_ns.append(avg_l_n)
        avg_l_ds.append(avg_l_d)

# Save model after last epoch
if save_models:
    print("saving model")
    model_fn = os.path.join(model_dir, model_name)
    torch.save(TileNet.state_dict(), model_fn)
 
with open(results_path / str(model_name + ".pkl"), "wb") as f:
    avg = {"losses": avg_losses, "l_n": avg_l_ns, "l_d": avg_l_ds}
    pickle.dump(avg, f)
