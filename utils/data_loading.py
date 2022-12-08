import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image,ImageChops
from torch.utils.data import Dataset
import random
import torchvision.transforms as T

class SatelliteDataset(Dataset):
    def __init__(self, preimages_dir,premasks_dir,images_dir, masks_dir, scale: float = 1.0, increase = 0, values = [[1,1], False, False,False], probabilities = [0,0,0,0], mask_suffix: str = '',seed = 42,normalizemodel = 'vgg19'):
        random.seed(seed)
        self.normalize = normalizemodel
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.preimages_dir = Path(preimages_dir)
        self.premasks_dir = Path(premasks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.values = values
        self.probabilities = probabilities
        self.mask_suffix = mask_suffix
        self.ids = [file.split('_')[0:2] for file in listdir(images_dir) if not file.startswith('.')]
        if(increase > 0):
            copyid = self.ids
            newid = []
            for i in range(increase + len(self.ids)):
                ranint = random.randint(0,len(self.ids) - 1)
                newid.append(copyid[ranint])
            self.ids = newid
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale = [1,1], flip = False, rotate = False,is_mask = False, randlist = [0,0,0,0], roll = False):
            w, h = pil_img.size            
            if(scale!=[1,1]):
                scalediff = randlist[0] * (scale[1] - scale[0])
                newscale = scale[0] + scalediff
                newW, newH = int(newscale * w), int(newscale * h)
                left = int((newW - w)/2)
                top = int((newH - h)/2)
                right = int((w + newW)/2)
                bottom = int((h + newH)/2)
                if(newW > w):
                    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
                    pil_img = pil_img.crop((left, top, right, bottom))
                    pil_img = pil_img.resize((w, h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
                if(newW < w):
                    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
                    if(is_mask):
                        img_new = Image.new('L', (w,h), (0))
                        img_new.paste(pil_img, (int(-1 * left), int(-1 * top)))
                        pil_img = img_new
                    else:
                        img_new = Image.new('RGB', (w,h), (0,0,0))
                        img_new.paste(pil_img, (int(-1 * left), int(-1 * top)))
                        pil_img = img_new
            if(flip == True):
                if(randlist[1] > 0.5):
                    pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                if(randlist[2] > 0.5):
                    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            if(rotate == True):
                chance = randlist[3]
                pil_img = pil_img.rotate(chance*90,resample = Image.NEAREST if is_mask else Image.BICUBIC)
            
            if(roll == True):
                chance1 = randlist[4]
                chance2 = randlist[5]
                pil_img = ImageChops.offset(pil_img, int(chance1 * 512), int(chance2 *512))
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
        mask_file = [str(self.masks_dir) + '\\' +name[0] + '_' + name[1] + '_post_disaster'+ self.mask_suffix]
        premask_file = [str(self.premasks_dir) + '\\' +name[0] + '_' + name[1] + '_pre_disaster'+ self.mask_suffix]
        img_file = [str(self.images_dir) + '\\' +name[0] + '_' + name[1]+ '_post_disaster'+self.mask_suffix]
        preimg_file = [str(self.preimages_dir) + '\\' +name[0] + '_' + name[1]  + '_pre_disaster'+self.mask_suffix]
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(preimg_file) == 1, f'Either no image or multiple images found for the ID {name}: {preimg_file}'
        assert len(premask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {premask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        premask = self.load(premask_file[0])
        preimg = self.load(preimg_file[0])
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        assert preimg.size == premask.size, \
            f'Image and mask {name} should be the same size, but are {preimg.size} and {premask.size}'
        prearray = [[1,1], False, False,False]
        if(random.random() < self.probabilities[0]):
            prearray[0] = self.values[0]
        if(random.random() < self.probabilities[1]):
            prearray[1] = self.values[1]
        if(random.random() < self.probabilities[2]):
            prearray[2] = self.values[2]
        if(random.random() < self.probabilities[3]):
            prearray[3] = self.values[3]    
        randlist = [random.random(),random.random(),random.random(),random.randrange(4),random.random(),random.random()]
        img = self.preprocess(img,
                              randlist = randlist,
                              scale = prearray[0], 
                              flip = prearray[1],
                              rotate = prearray[2],
                              roll = prearray[3],
                              is_mask=False,
                              )
        mask = self.preprocess(mask,
                              randlist = randlist,
                              scale = prearray[0], 
                              flip = prearray[1],
                              rotate = prearray[2],
                              roll = prearray[3],
                              is_mask=True)
        preimg = self.preprocess(preimg,
                              randlist = randlist,
                              scale = prearray[0], 
                              flip = prearray[1],
                              rotate = prearray[2],
                              roll = prearray[3],
                              is_mask=False)
        premask = self.preprocess(premask,
                              randlist = randlist,
                              scale = prearray[0], 
                              flip = prearray[1],
                              rotate = prearray[2],
                              roll = prearray[3],
                              is_mask=True)
        if (self.normalize =='vgg19'):
            transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            return {
                'image': transform(torch.as_tensor(img.copy()).float().contiguous()),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'preimage': transform(torch.as_tensor(preimg.copy()).float().contiguous()),
                'premask': torch.as_tensor(premask.copy()).long().contiguous()
            }
        else:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                'preimage': torch.as_tensor(preimg.copy()).float().contiguous(),
                'premask': torch.as_tensor(premask.copy()).long().contiguous()
            }
