import tifffile as tiff
import numpy as np
import PIL
import os
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def createlabels(inputfol,outputpostfol,outputprefol,size):
    for i in range(len(inputfol)):
        inputpath = inputfol[i]
        outputpost = outputpostfol[i]
        outputpre = outputprefol[i]
        for root, dirs, files in os.walk(inputpath, topdown=False):
            with tqdm(total=len(files), desc='Saving to ' + outputpost + ' and ' + outputpre, unit='img') as pbar:
                for name in files:
                    if('pre_disaster' in name):
                        outfile = os.path.splitext(os.path.join(outputpre, name))[0] + ".png"
                        im = tiff.imread(os.path.join(root, name))
                        ima = im.astype(dtype=np.uint8)
                        imarray = Image.fromarray(ima)
                        imarray = imarray.resize(size,PIL.Image.NEAREST)
                        imarray.save(outfile, "PNG", quality=100)
                    if('post_disaster' in name):
                        outfile = os.path.splitext(os.path.join(outputpost, name))[0] + ".png"
                        im = tiff.imread(os.path.join(root, name))
                        ima = im.astype(dtype=np.uint8)
                        imarray = Image.fromarray(ima)
                        imarray = imarray.resize(size,PIL.Image.NEAREST)
                        imarray.save(outfile, "PNG", quality=100)
                    pbar.update(1)