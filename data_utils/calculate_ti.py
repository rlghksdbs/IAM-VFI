import os
import argparse
import cv2
from tqdm import tqdm

def CreateDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("make directory: "+path)
    else:
        print(path + " already exsisted")

        
def ti_calc(frame, previous_frame): #input frame: gray_scale
    frame = frame.astype("float")
    value = 0
    if previous_frame is not None:
        value = (frame - previous_frame).std()
    return value
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputpath', type=str, default='../patch_data/')
    parser.add_argument('--tipath', type=str, default='../patch_data/')
    args = parser.parse_args()

    inputpath = args.inputpath
    tipath = args.tipath

    train_fn = os.path.join(inputpath, 'all_patch.txt')

    with open(train_fn, 'r') as f:
        train_list = f.read().splitlines()

    train_len = len(train_list)

    CreateDirectory(tipath)
    
    origin_ti_max = []

    ti_max = open(os.path.join(tipath, 'patch_ti.txt'), 'w')

    for i in tqdm(range(train_len), desc='calculate_patch_TI'):
        input_clip = os.path.join(inputpath, train_list[i])
    
        input_frame = sorted(os.listdir(input_clip))

        img1_path = os.path.join(input_clip, input_frame[0])
        img2_path = os.path.join(input_clip, input_frame[1])
        img3_path = os.path.join(input_clip, input_frame[2])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img3 = cv2.imread(img3_path)
        
        ti = ti_calc(img1, img3)
        ti_max_val = ti

        origin_ti_max.append(ti_max_val)

        ti_max.write('{}\t {}\n'.format(train_list[i], ti))