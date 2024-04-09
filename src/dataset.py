
import rasterio
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset,  DataLoader
from segment_anything import SamPredictor, sam_model_registry
from transformers import SamProcessor
import numpy as np


def simple_equalization_8bit(im, percentiles=5):
    ''' im is a numpy array
        returns a numpy array
    '''
    out = np.zeros_like(im)
    # faire l'equalization par channel
    def equalize(im_channel):
        v_min, v_max = np.percentile(im_channel,percentiles),np.percentile(im_channel, 100 - percentiles)

        # Clip the image to the percentile values
        im_clipped = np.clip(im_channel, v_min, v_max)

        # Scale the image to the 0-255 range
        im_scaled = np.round((im_clipped - v_min) / (v_max - v_min))
        return im_scaled.astype(np.uint8)
    
    for channel in range(im.shape[0]):
        out[channel,:,:] = equalize(im[channel,:,:])
    
    return out

def calculate_ndwi(img):
    # Extract bands
    green_band = img[1,:,:]
    nir_band = img[3,:,:]
    
    ndwi = (nir_band.astype(float) - green_band.astype(float)) / (nir_band + green_band)
    ndwi = np.nan_to_num(ndwi, nan=0.0)

    return ndwi

class S1S2Dataset(Dataset):
    def __init__(self, img_folder, mask_folder, processor, transform=None, target_transform=None, ndwi=False):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.target_transform = target_transform
        self.img_filenames = [f for f in os.listdir(img_folder) if f.endswith('.tif') and "img" in f]
        self.processor = processor
        self.ndwi = ndwi

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_folder, filename)
        
        # image
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            if self.ndwi:
                ndwi = calculate_ndwi(image)
                # replace channel 1 with ndwi
                image[1,:,:] = ndwi
            image = image[0:3, :, :]
                
            image = simple_equalization_8bit(image, percentiles=5) 
            image = torch.from_numpy(image) # shape (C, H, W)
            
            image = self.processor(image, return_tensors="pt")
            # remove batch dimension
            image = {k: v.squeeze(0) for k, v in image.items()}

        # masque
        mask_filename = filename.replace("img", "msk")
        mask_path = os.path.join(self.mask_folder, mask_filename)
        with rasterio.open(mask_path) as src:
            mask = src.read()[0].astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)  # Ajouter une dimension de canal (C, H, W)

        return image, mask
    

if __name__ == "__main__":
    
    data_dir = "split3"

    train_img_dir = os.path.join(data_dir, 'train', 'img')
    test_img_dir = os.path.join(data_dir, 'val', 'img')
    train_msk_dir = os.path.join(data_dir, 'train', 'msk')
    test_msk_dir = os.path.join(data_dir, 'val', 'msk')
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    dataset = S1S2Dataset(train_img_dir, train_msk_dir, processor=processor)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("train_loader:", len(train_loader))
    print(next(iter(train_loader))[0]["pixel_values"].shape)