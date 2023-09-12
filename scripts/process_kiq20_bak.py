import os
import scipy.io as sio
import numpy as np
from PIL import Image
import pickle
import csv
from glob import glob
from tqdm import tqdm
import random
import re

debug = 0
num_images = 105

def get_meta_info(seed=123):
    """Generate meta information and train/val/test splits for KIQ20 dataset.

    """
    all_label_file = '../datasets/KK_dataset/kankan_scores.csv'
    save_meta_path = '../datasets/meta_info/meta_info_KIQ20Dataset.csv'
    split_info = {'train': [], 'val': [], 'test': []}

    raw_files = glob("../datasets/KK_dataset/KIQ20_images/*.jpg")
    
    # separate 1% as validation part
    split_info['train'] = list(range(num_images))
    random.seed(seed)
    train_split = split_info['train']
    # Fixed 105 images
    sep_idx = int(round(num_images * 0.8))
    sep_idx2 = int(round(num_images * 0.1))
    if debug: print(f"sep idx: {sep_idx}, sep_idx2: {sep_idx2}")

    random.shuffle(train_split)
    if debug: print(f"train split: {len(train_split)} -> {train_split}")
    
    split_info['train'] = train_split[:sep_idx]
    split_info['val'] = train_split[sep_idx:(sep_idx+sep_idx2)]
    split_info['test'] = train_split[(sep_idx+sep_idx2):]
    if debug: print(f"After split:\n\t{split_info['train']}\n\t{split_info['val']}\n\t{split_info['test']}")
    
    with open(all_label_file, 'r', encoding='utf-8') as f, open(save_meta_path, 'w+', newline='') as sf:
        csvreader = csv.reader(f)
        head = next(csvreader)
        if debug: print(f"head: {head}")

        csvwriter = csv.writer(sf)
        new_head = ['img_name', 'sharpness', 'color', 'brightness', 'noisiness', 'average', 'split']
        csvwriter.writerow(new_head)
        for idx, row in enumerate(csvreader):
            for name in raw_files:
                _, name = os.path.split(name)
                if re.match('^'+row[0], name) != None:
                    row[0] = name if os.path.exists(os.path.join(_, name)) else print(f"Match error name ===> {name}")
                    break
                    print(f"You should not see me!!!!")
            split_info['train'].append(idx)
            new_row = row[:6] + ['train']
            if idx in split_info['val']:
                new_row[6] = 'val'
            elif idx in split_info['test']:
                new_row[6] = 'test'
            
            if debug: print(new_row)
            csvwriter.writerow(new_row)
    save_split_path = '../datasets/meta_info/KIQ20Dataset.pkl'
    with open(save_split_path, 'wb') as sf:
        pickle.dump({1: split_info}, sf)

if __name__ == '__main__':
    get_meta_info()
    #  get_random_splits()
