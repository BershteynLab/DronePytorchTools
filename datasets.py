from pathlib import Path
import sys
from PIL import Image
import json
import random
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset



def get_annotations(anno_path):
    df = pd.read_csv(anno_path, header=None, names=["x1","y1","x2","y2","x3","y3","x4","y4","_","_2"], sep=" ")
    df.loc[:,"box_x1"] = df.loc[:,["x1","x2","x3","x4"]].min(axis=1)
    df.loc[:,"box_x2"] = df.loc[:,["x1","x2","x3","x4"]].max(axis=1)
    df.loc[:,"box_y1"] = df.loc[:,["y1","y2","y3","y4"]].min(axis=1)
    df.loc[:,"box_y2"] = df.loc[:,["y1","y2","y3","y4"]].max(axis=1)
    return df.loc[:,["box_y1","box_x1","box_y2","box_x2"]].values

def get_dots(anno):
    ys = (anno[:,0] + anno[:,2])/2
    xs = (anno[:,1] + anno[:,3])/2
    return np.stack([xs,ys], axis=-1)

class ValData(Dataset):
    def __init__(self, data_path, anno_dir, data_split_file, im_dir, split='val', crop_size = 100):
        anno_dir = Path(data_path) / anno_dir

        with open(Path(data_path) / data_split_file) as f:
            data_split = json.load(f)

        self.img = data_split[split]
        random.shuffle(self.img)
        self.split = split
        self.crop_size = crop_size
        self.im_dir = Path(data_path) / im_dir
        self.tiles_per_image = (int(1000/crop_size) \
                                 if (1000/crop_size) == int(1000/crop_size) \
                                 else int(1000/crop_size) + 1) ** 2
        self.annotations = {img: get_annotations(anno_dir / (img + ".txt")) for img in self.img}
        self.dots = {img: get_dots(self.annotations[img]) for img in self.annotations}

    def __len__(self):
        return len(self.img) * self.tiles_per_image

    def __getitem__(self, idx):
        im_id = self.img[idx // self.tiles_per_image]
        pane = idx % self.tiles_per_image
        n_col_row = int(round(self.tiles_per_image**0.5))
        row = pane // n_col_row
        col = pane % n_col_row
        dots = self.dots[im_id]
        image = Image.open(f'{self.im_dir}/{ im_id }.png')
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.load()
        resized_density = np.zeros((1000, 1000), dtype='float32')

        for i in range(dots.shape[0]):
            resized_density[min(int(dots[i][1]),999),min(int(dots[i][0]), 999)] = 1
        image = transforms.ToTensor()(image)
        starts = list(range(0, 1000 - self.crop_size, self.crop_size)) + [1000 - self.crop_size]
        start_H = starts[row]
        start_W = starts[col]
        resized_density = ndimage.gaussian_filter(resized_density, sigma=8, radius=16, order=0) * 60
        reresized_image = TF.crop(image, start_H, start_W, self.crop_size, self.crop_size)
        reresized_density = resized_density[start_H: start_H+self.crop_size, start_W:start_W + self.crop_size] 

        # Gaussian distribution density map
        return (reresized_image, reresized_density) 
