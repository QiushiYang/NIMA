#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:45:57 2018

@author: qsyang
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils import data

class AVADataset(data.Dataset):
    """AVA dataset
    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir,transform=None):
        self.annotations = np.loadtxt(csv_file,'int') # 'int'
        self.root_dir = root_dir
        self.transform = transform
#         self.style_ann = np.loadtxt(style_file,'int')

    def __len__(self):
        return len(self.annotations)-1

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations[idx, 1]) + '.jpg')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_name)
        image = image.convert("RGB")
        
#         x,y,h = np.shape(image)
#         style = np.zeros([x,y,14])
#         array = np.array(image)       
#         style_ann = self.style_ann[idx,1]       
#         style[:,:,style_ann] = 1        
# #        img = np.row_stack([array,style])
#         img = np.concatenate((array, style), axis = 2) 

        annotations = self.annotations[idx, 2:12]
        num = annotations.sum(axis=0)
        annotations = annotations / sum(annotations)
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations, 'number': num}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
