import numpy as np
from osgeo import gdal
import os
import random

def load_img(img_file, val_type='uint8', bands_only=False, num_bands=4):
    """
    Loads an image using gdal, returns it as an array.
    """
    obj = gdal.Open(img_file)
    if val_type == 'uint8':
        img = obj.ReadAsArray().astype(np.uint8)
    elif val_type == 'float32':
        img = obj.ReadAsArray().astype(np.float32)
    else:
        raise ValueError('Invalid val_type for image values. Try uint8 or float32.')
    img = np.moveaxis(img, 0, -1)
    if bands_only: img = img[:,:,:num_bands]
    return img

# TO CHANGE
def load_img_custom_bands(img_file, val_type='uint8', bands_only=False):

    # COLOR INFRATED
    # selected_bands = [7, 3, 2]
    
    # AGRICULTURE
    # selected_bands = [11, 7, 1]
    
    # VEGETATION INDEX
    ## vegetation_bands = (7 - 3)/(7 + 3)
    
    # MOISTURE INDEX
    ## moisture_bands = (8 - 11)/(8 + 11)

    obj = gdal.Open(img_file)
    if obj is None:
        raise FileNotFoundError(f"File {img_file} not found")
    
    if val_type == 'uint8':
        img = obj.ReadAsArray().astype(np.uint8)
    elif val_type == 'float32':
        img = obj.ReadAsArray().astype(np.float32)
    else:
        raise ValueError('Invalid val_type for image values. Try uint8 or float32.')
    
    img = np.moveaxis(img, 0, -1)  # Move the bands axis to the last dimension

    if bands_only:
        
        # COLOR INFRATED/AGRICULTURE
        # img = img[:, :, selected_bands]
        
        # VEGETATION INDEX
        # img[:, :, vegetation_bands] = (img[:,:,7] - img[:,:,3]) / (img[:,:,7] + img[:,:,3])
        # img = img[:, :, vegetation_index]
        
        # MOISTURE INDEX
        img[:, :, moisture_bands] = (img[:,:,7] - img[:,:,3]) / (img[:,:,7] + img[:,:,3])
        img = img[:, :, moisture_index]
        
    
    return img



def get_triplet_imgs(img_dir, img_ext='.tif', n_triplets=1000):
    """
    Returns a numpy array of dimension (n_triplets, 2). First column is
    the img name of anchor/neighbor tiles and second column is img name 
    of distant tiles.
    """
    img_names = []
    for filename in os.listdir(img_dir):
        if filename.endswith(img_ext):
            img_names.append(filename)
    img_triplets = list(map(lambda _: random.choice(img_names), range(2 * n_triplets)))
    img_triplets = np.array(img_triplets)
    return img_triplets.reshape((-1, 2))

def get_triplet_imgs_with_dirs(img_dir_list, img_ext='.tif', n_triplets=1000):
    """
    Returns a numpy array of dimension (n_triplets, 2). First column is
    the img name of anchor/neighbor tiles and second column is img name 
    of distant tiles.
    """
    img_names = []
    for img_dir in img_dir_list:
        for filename in os.listdir(img_dir):
            if filename.endswith(img_ext):
                img_names.append(os.path.basename(img_dir)+"/"+filename)
    img_triplets = list(map(lambda _: random.choice(img_names), range(2 * n_triplets)))
    img_triplets = np.array(img_triplets)
    return img_triplets.reshape((-1, 2))

def get_triplet_tiles(tile_dir, img_dir, img_triplets, tile_size=50, neighborhood=100, 
                      val_type='uint8', bands_only=False, save=True, verbose=False):
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2

    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    tiles = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    for img_name in unique_imgs:
        print("Sampling image {} from dir".format(img_name))
        if img_name[-3:] == 'npy':
            img = np.load(os.path.join(img_dir, img_name))
        else:
            img = load_img_custom_bands(os.path.join(img_dir, img_name), val_type=val_type, 
                       bands_only=bands_only)
        img_padded = np.pad(img, pad_width=[(tile_radius, tile_radius),
                                            (tile_radius, tile_radius), (0,0)],
                            mode='reflect')
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
                        tile_anchor = tile_anchor[:-1,:-1]
                        tile_neighbor = tile_neighbor[:-1,:-1]
                    np.save(os.path.join(tile_dir, '{}anchor.npy'.format(idx)), tile_anchor)
                    np.save(os.path.join(tile_dir, '{}neighbor.npy'.format(idx)), tile_neighbor)
                
                tiles[idx,0,:] = xa - tile_radius, ya - tile_radius
                tiles[idx,1,:] = xn - tile_radius, yn - tile_radius
                
                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    xd, yd = sample_distant_same(img_shape, xa, ya, neighborhood, tile_radius)
                    if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((xd, yd)))
                    if save:
                        tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                        if size_even:
                            tile_distant = tile_distant[:-1,:-1]
                        np.save(os.path.join(tile_dir, '{}distant.npy'.format(idx)), tile_distant)
                    tiles[idx,2,:] = xd - tile_radius, yd - tile_radius
            
            elif row[1] == img_name: 
                # distant image is different from anchor/neighbor image
                xd, yd = sample_distant_diff(img_shape, tile_radius)
                if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((xd, yd)))
                if save:
                    tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                    if size_even:
                        tile_distant = tile_distant[:-1,:-1]
                    np.save(os.path.join(tile_dir, '{}distant.npy'.format(idx)), tile_distant)
                tiles[idx,2,:] = xd - tile_radius, yd - tile_radius
            
    return tiles

def sample_anchor(img_shape, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xa = np.random.randint(0, w) + tile_radius
    ya = np.random.randint(0, h) + tile_radius
    return xa, ya

def sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xn = np.random.randint(max(xa-neighborhood, tile_radius),
                           min(xa+neighborhood, w+tile_radius))
    yn = np.random.randint(max(ya-neighborhood, tile_radius),
                           min(ya+neighborhood, h+tile_radius))
    return xn, yn


def sample_distant_same(img_shape, xa, ya, neighborhood, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xd, yd = xa, ya
    if (xa + neighborhood >= w and xa - neighborhood <= 0) or yd + neighborhood >= h and yd - neighborhood <= 0:
        raise ValueError("Image is to small to sample two images")
    
    while (xd >= xa - neighborhood) and (xd <= xa + neighborhood):
        xd = np.random.randint(0, w) + tile_radius
    while (yd >= ya - neighborhood) and (yd <= ya + neighborhood):
        yd = np.random.randint(0, h) + tile_radius
    return xd, yd


def sample_distant_diff(img_shape, tile_radius):
    return sample_anchor(img_shape, tile_radius)

def extract_tile(img_padded, x0, y0, tile_radius):
    """
    Extracts a tile from a (padded) image given the row and column of
    the center pixel and the tile size. E.g., if the tile
    size is 15 pixels per side, then the tile radius should be 7.
    """
    w_padded, h_padded, c = img_padded.shape
    row_min = x0 - tile_radius
    row_max = x0 + tile_radius
    col_min = y0 - tile_radius
    col_max = y0 + tile_radius
    assert row_min >= 0, 'Row min: {}'.format(row_min)
    assert row_max <= w_padded, 'Row max: {}'.format(row_max)
    assert col_min >= 0, 'Col min: {}'.format(col_min)
    assert col_max <= h_padded, 'Col max: {}'.format(col_max)
    tile = img_padded[row_min:row_max+1, col_min:col_max+1, :]
    return tile
