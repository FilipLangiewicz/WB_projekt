import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
import random

from sample_tiles import (
    extract_tile,
    sample_distant_same,
    sample_neighbor,
    load_img_color_infrated,
    sample_anchor,
    sample_distant_diff,
)
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append("..")
import pandas as pd
from pathlib import Path
from osgeo import gdal

# script creates the triplets to train the tile2vec model

img_type = "landsat"  # images are in float - this parameter specifies that there is a need for normalization of floats
tile_dir = Path("/storage/tile2vec/limited_bands/test")
base_eurosat_dir = Path("/storage/EuroSATallBands")
bands = 3
num_workers = 4
n_triplets = 50000 # number of triplets to sample

train_path = Path("/storage/EuroSATallBands/train.csv")
train_df = pd.read_csv(train_path)

random.seed(44)
np.random.seed(44)


# modified function by us
def get_triplet_imgs(img_df, n_triplets=1000):
    """
    Returns a numpy array of dimension (n_triplets, 2). First column is
    the img name of anchor/neighbor tiles and second column is img name
    of distant tiles.
    """
    img_names = []
    img_names = img_df["Filename"].to_numpy()
    print("Sampling tiles")
    a = [
        np.random.choice(img_names, size=2, replace=False)
        for _ in tqdm(range(n_triplets))
    ]
    img_triplets = np.vstack(a)
    return img_triplets


# modified function by us
def get_triplet_tiles(
    tile_dir,
    img_dir,
    img_triplets,
    tile_size=50,
    neighborhood=100,
    val_type="uint8",
    bands_only=False,
    save=True,
    verbose=False,
):
    print("Loading and preprocessing triplets")
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    size_even = tile_size % 2 == 0
    tile_radius = tile_size // 2

    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    tiles = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    for img_name in tqdm(unique_imgs):
        if img_name[-3:] == "npy":
            img = np.load(os.path.join(img_dir, img_name))
        else:
            img = load_img_color_infrated(
                os.path.join(img_dir, img_name),
                val_type=val_type,
                bands_only=bands_only,
            )
        img_padded = np.pad(
            img,
            pad_width=[(tile_radius, tile_radius), (tile_radius, tile_radius), (0, 0)],
            mode="reflect",
        )
        img_shape = img_padded.shape

        for idx, row in enumerate(img_triplets):
            if row[0] == img_name:
                xa, ya = sample_anchor(img_shape, tile_radius)
                xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)

                if verbose:
                    print("    Saving anchor and neighbor tile #{}".format(idx))
                    print("    Anchor tile center:{}".format((xa, ya)))
                    print("    Neighbor tile center:{}".format((xn, yn)))
                if save:
                    tile_anchor = extract_tile(img_padded, xa, ya, tile_radius)
                    tile_neighbor = extract_tile(img_padded, xn, yn, tile_radius)
                    if size_even:
                        tile_anchor = tile_anchor[:-1, :-1]
                        tile_neighbor = tile_neighbor[:-1, :-1]
                    np.save(
                        os.path.join(tile_dir, "{}anchor.npy".format(idx)), tile_anchor
                    )
                    np.save(
                        os.path.join(tile_dir, "{}neighbor.npy".format(idx)),
                        tile_neighbor,
                    )

                tiles[idx, 0, :] = xa - tile_radius, ya - tile_radius
                tiles[idx, 1, :] = xn - tile_radius, yn - tile_radius

                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    try:
                        xd, yd = sample_distant_same(
                            img_shape, xa, ya, neighborhood, tile_radius
                        )
                    except ValueError:
                        print("Could not sample from the same image")
                        print("Image name ", img_name)
                        print("Exiting...")
                        exit(0)

                    if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((xd, yd)))
                    if save:
                        tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                        if size_even:
                            tile_distant = tile_distant[:-1, :-1]
                        np.save(
                            os.path.join(tile_dir, "{}distant.npy".format(idx)),
                            tile_distant,
                        )
                    tiles[idx, 2, :] = xd - tile_radius, yd - tile_radius

            elif row[1] == img_name:
                # distant image is different from anchor/neighbor image
                xd, yd = sample_distant_diff(img_shape, tile_radius)
                if verbose:
                    print("    Saving distant tile #{}".format(idx))
                    print("    Distant tile center:{}".format((xd, yd)))
                if save:
                    tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                    if size_even:
                        tile_distant = tile_distant[:-1, :-1]
                    np.save(
                        os.path.join(tile_dir, "{}distant.npy".format(idx)),
                        tile_distant,
                    )
                tiles[idx, 2, :] = xd - tile_radius, yd - tile_radius

    return tiles


in_channels = bands
z_dim = 512

img_triplets = get_triplet_imgs(train_df, n_triplets)
print("finished generating triplet sources")

tiles = get_triplet_tiles(tile_dir, base_eurosat_dir, img_triplets, tile_size=60)
