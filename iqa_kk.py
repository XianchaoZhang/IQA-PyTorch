import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import pandas as pd
import numpy as np

def main():
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../dataset', help='input image/folder path.')
    parser.add_argument('-m', '--metric_name', type=str, default='clipiqa', help='IQA metric name, case sensitive.')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursive search or not.')

    args = parser.parse_args()

    metric_name = args.metric_name.lower()

    # set up IQA model
    iqa_model = create_metric(metric_name)

    p = '*.jpg' #[jpg, jpeg, bmp, png]'
    if args.recursive:
        p = '**/' + p
    #print(f"p: {p}")

    if os.path.isfile(args.input):
        input_paths = [args.input]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.input, p), recursive=args.recursive))
        #print(f"input_paths: {input_paths}")
        #exit(0)

    attrs = ['Quality', 'Brightness', 'Contrast', 'Sharpness', 'SharpEdge', 'Resolution', 'Noiseness']
    attr_rows = []
    fp_rows = []
    fn_row = []
    test_img_num = len(input_paths)
    pbar = tqdm(total=test_img_num, unit='image')
    for idx, img_path in enumerate(input_paths):
        fp, fn = os.path.split(img_path)
        score = iqa_model(img_path).cpu().numpy()[0]
        score = np.around(score*100, 1)
        attr_rows.append(score)
        fp_rows.append(fp)
        fn_row.append(fn)
        #print(score)
        #exit(0)
        pbar.update(1)
        #pbar.set_description(f'desc {metric_name} of {fn}: {score}')
        #pbar.write(f'{metric_name} of {fn}: {score}')

    pbar.close()
    #print(f"fp_rows: {fp_rows}")
    out_df = pd.DataFrame(attr_rows, columns=attrs, index=fn_row)
    out_df.to_csv("iqa.csv")
    print(f'Done!')

if __name__ == '__main__':
    main()
