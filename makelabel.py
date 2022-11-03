import numpy as np
import PIL
import os
import glob
import json
from geomet import wkt
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def createlabels(inputfol,outputpostfol,outputprefol,size):
    for i in range(len(inputfol)):
        inputpath = inputfol[i]
        outputpost = outputpostfol[i]
        outputpre = outputprefol[i]
        json_list = []
        for filename in os.listdir(inputpath):
            if filename.endswith('.json'):
                data = open(inputpath + filename)
                json_list.append((json.load(data),filename))
        with tqdm(total=len(json_list), desc='Saving to ' + outputpost + ' and ' + outputpre, unit='img') as pbar:
            for js in json_list:
                width = js[0]['metadata']['width']
                height = js[0]['metadata']['height']
                image = Image.new("L", (width, height), color = (0))
                draw = ImageDraw.Draw(image)
                if('pre_disaster' in js[1]):
                    for j in js[0]['features']['xy']:
                        polygonlist = wkt.loads(j['wkt'])
                        polygontuples = [(x,y) for x,y in polygonlist['coordinates'][0]]
                        draw.polygon((polygontuples), fill=(1))
                    image.save(outputpre+js[1].split('.')[0] + '.png')
                if('post_disaster' in js[1]):
                    for j in js[0]['features']['xy']:
                        subtype = j['properties']['subtype']
                        fill = (0)
                        polygonlist = wkt.loads(j['wkt'])
                        polygontuples = [(x,y) for x,y in polygonlist['coordinates'][0]]
                        match subtype:
                            case "no-damage":
                                fill = (1)
                            case "minor-damage":
                                fill = (2)
                            case "major-damage":
                                fill = (3)
                            case "destroyed":
                                fill = (4)
                            case _:
                                fill = (0)
                        draw.polygon((polygontuples), fill=fill)
                    image = image.resize(size,PIL.Image.NEAREST)
                    image.save(outputpost+js[1].split('.')[0] + '.png')
                pbar.update(1)