import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random

class BasicDataset(Dataset):
    def __init__(self, images_dir, masks_dir, scale: float = 1.0, increase = 0, values = [1, False, False, 0, None, 0, 0], probabilities = [0,0,0,0,0,0,0], mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.values = values
        self.probabilities = probabilities
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if(increase > 0):
            copyid = self.ids
            newid = []
            for i in range(increase + len(self.ids)):
                ranint = random.randint(0,len(self.ids) - 1)
                newid.append(copyid[ranint])
            self.ids = newid
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale = 1, flip = False, rotate = False,noise = 0, color_space = None, contrast = 0, sharpen = 0,is_mask = False):
            chance = random.random()
            chance2 = random.random()
            w, h = pil_img.size
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            if(scale!=1):
                pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
            if(flip == True):
                if(chance > 0.5):
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            if(rotate == True):
                pil_img = pil_img.rotate(chance*360,resample = Image.NEAREST if is_mask else Image.BICUBIC)
            if(noise !=0 and is_mask !=True):
                noise = np.random.normal(0, noise, (newH, newW))
                np_pil_img = np.array(pil_img)
                npmask = (np_pil_img == 0)
                np_pil_img[:,:,0] = np_pil_img[:,:,0] + noise
                np_pil_img[:,:,1] = np_pil_img[:,:,1] + noise
                np_pil_img[:,:,2] = np_pil_img[:,:,2] + noise
                np_pil_img[npmask] = 0
                pil_img = Image.fromarray(np.uint8(np_pil_img))
            if(color_space != None and is_mask !=True):
                np_pil_img = np.array(pil_img)
                r = np_pil_img[:,:,color_space[0]]
                g = np_pil_img[:,:,color_space[1]]
                b = np_pil_img[:,:,color_space[2]]
                np_pil_img[:,:,0] = r
                n_ppil_img[:,:,1] = g
                np_pil_img[:,:,2] = b
                pil_img = Image.fromarray(np.uint8(np_pil_img))
            if(contrast != 0 and is_mask !=True):
                contrast = ImageEnhance.Contrast(pil_img)
                pil_img = pil_img.enhance(contrast)
            if(sharpen != 0 and is_mask !=True):
                for i in range(sharpen):
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
            
            img_ndarray = np.asarray(pil_img)
    
            if not is_mask:
                if img_ndarray.ndim == 2:
                    img_ndarray = img_ndarray[np.newaxis, ...]
                else:
                    img_ndarray = img_ndarray.transpose((2, 0, 1))
    
                img_ndarray = img_ndarray / 255
    
            return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        print(self.masks_dir)
        mask_file = [str(self.masks_dir) + '\\' +name + self.mask_suffix]
        img_file = [str(self.images_dir) + '\\' +name + self.mask_suffix]

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        print(mask_file[0])
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        prearray = [1, False, False, 0, None, 0, 0]
        cprobsum = np.cumsum(self.probabilities)
        chance = random.random()
        if(chance < cprobsum[0]):
            prearray[0] = self.values[0]
        elif(chance < cprobsum[1]):
            prearray[1] = self.values[1]
        elif(chance < cprobsum[2]):
            prearray[2] = self.values[2]
        elif(chance < cprobsum[3]):
            prearray[3] = self.values[3]
        elif(chance < cprobsum[4]):
            prearray[4] = self.values[4]
        elif(chance < cprobsum[5]):
            prearray[5] = self.values[5]
        elif(chance < cprobsum[6]):
            prearray[6] = self.values[6]            

        img = self.preprocess(img, 
                              scale = prearray[0], 
                              flip = prearray[1],
                              rotate = prearray[2],
                              noise = prearray[3], 
                              color_space = prearray[4],
                              contrast = prearray[5], 
                              sharpen = prearray[6],
                              is_mask=False)
        mask = self.preprocess(mask,
                              scale = prearray[0], 
                              flip = prearray[1],
                              rotate = prearray[2],
                              noise = prearray[3], 
                              color_space = prearray[4],
                              contrast = prearray[5], 
                              sharpen = prearray[6],
                              is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

