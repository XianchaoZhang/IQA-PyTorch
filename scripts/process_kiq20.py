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

debug = 1
num_images = 21

def get_meta_info(seed=123):
    """Generate meta information and train/val/test splits for KIQ20 dataset.

    """
    all_label_file = '../datasets/KK_dataset/kankan_scores.csv'
    save_meta_path = '../datasets/meta_info/meta_info_KIQ20Dataset_text.csv'
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
        new_head = ['img_name', 'average', 'sharpness', 'color', 'brightness', 'noisiness', 'split']
        csvwriter.writerow(new_head)
        for idx, row in enumerate(csvreader):
            for name in raw_files:
                _, name = os.path.split(name)
                if re.match('^'+row[0], name) != None:
                    row[0] = name if os.path.exists(os.path.join(_, name)) else print(f"Match error name ===> {name}")
                    break
                    print(f"You should not see me!!!!")
            split_info['train'].append(idx)
            new_row = [row[0]] + [row[5]] + row[1:5] + ['train']
            if idx in split_info['val']:
                new_row[6] = 'val'
            elif idx in split_info['test']:
                new_row[6] = 'test'
            
            if debug: print(new_row)
            csvwriter.writerow(new_row)
    
    
def get_split_phone(seed=123):
    all_label_file = '../datasets/meta_info/meta_info_KIQ20Dataset_text.csv'
    tmp_meta_path = '../datasets/meta_info/meta_info_KIQ20Dataset_'
    phones = ['Austion', 'Gaion', 'Galaxy A53', 'Galaxy S22+', 'Victoria']
    #split_info = {'train': [], 'val': [], 'test': []}

    for phone in phones:
        split_info = {'train': [], 'val': [], 'test': []}
        tmp_file = phone + ".csv"
        save_meta_path = tmp_meta_path + phone + '.csv'
        with open(all_label_file, 'r', encoding='utf-8') as f, open(tmp_file, 'w+', newline='') as sf:
            csvreader = csv.reader(f)
            head = next(csvreader)
            if debug: print(f"{phone} head: {head}")

            csvwriter = csv.writer(sf)
            csvwriter.writerow(head)
            i = 0
            for idx, row in enumerate(csvreader):
                if re.search(phone, row[0]) != None:
                    split_info['train'].append(i)
                    csvwriter.writerow(row)
                    i += 1
                else:
                    if debug: pass #print(f"tmp file: {row[0]} {phone} Not match")
        
        with open(tmp_file, 'r', encoding='utf-8') as f, open(save_meta_path, 'w+', newline='') as sf:
            csvreader = csv.reader(f)
            head = next(csvreader)

            csvwriter = csv.writer(sf)
            csvwriter.writerow(head)
            
            # separate 1% as validation part
            random.seed(seed)
            train_split = split_info['train']
            sep_idx = int(round(len(train_split) * 0.8))
            sep_idx2 = int(round(len(train_split) * 0.1))
            if debug: print(f"{__name__} len: {len(train_split)} sep idx: {sep_idx}, sep_idx2: {sep_idx2}")

            random.shuffle(train_split)
            if debug: print(f"{__name__} train split: {len(train_split)} -> {train_split}")
            
            split_info['train'] = train_split[:sep_idx]
            split_info['val'] = train_split[sep_idx:(sep_idx+sep_idx2)]
            split_info['test'] = train_split[(sep_idx+sep_idx2):]
            if debug: print(f"{__name__} After split:\n\t{split_info['train']}\n\t{split_info['val']}\n\t{split_info['test']}")
            for idx, row in enumerate(csvreader):
                row[6] = 0
                if idx in split_info['val']:
                    row[6] = 1
                elif idx in split_info['test']:
                    row[6] = 2
                csvwriter.writerow(row)
        # end per phone
        filename = os.path.splitext(save_meta_path)[0]
        filename = filename.replace('meta_info_', '')
        save_split_path = filename + '.pkl' 
        with open(save_split_path, 'wb') as sf:
            pickle.dump({1: split_info}, sf)

if __name__ == '__main__':
    get_meta_info()
    get_split_phone()

