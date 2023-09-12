import os
import csv
from glob import glob

debug = 1

def get_file_info():
    #all_label_file = '../datasets/KK_dataset/meta_info_KIQ20Dataset_text.csv'
    save_meta_path = '../datasets/meta_info/meta_info_KIQ100Dataset_text.csv'

    raw_files = glob("../datasets/KK_dataset/KIQ20_images/*.jpg")
      
    with open(save_meta_path, 'w+', newline='') as sf:
        csvwriter = csv.writer(sf)
        new_head = ['img_name', 'average', 'sharpness', 'color', 'brightness', 'noisiness', 'split']
        csvwriter.writerow(new_head)
        row = [''] * 7
        if debug: print(f"row: {row}")
        for name in raw_files:
            _, name = os.path.split(name)
            row[0] = name
            if debug: print(row)
            csvwriter.writerow(row)

if __name__ == '__main__':
    get_file_info()
    #  get_random_splits()
