import torch
import os
import cv2
import argparse
from tqdm import tqdm
def centercrop_ti(input_path, output_path):

    ti_train_tx = os.path.join(input_path, 'all.txt')

    with open(ti_train_tx, 'r') as f:
        train_list = f.read().splitlines()
    
    print(len(train_list))

    trainlist = train_list

    patch_train_txt = open(os.path.join(output_path, 'all_patch.txt'), 'w')

    for i in tqdm(range(len(trainlist))):
        data_path = os.path.join('../data', trainlist[i])

        out_path = os.path.join(output_path, trainlist[i])
        os.makedirs(out_path, exist_ok=True)

        if trainlist[i].startswith("01"):
            img1 = cv2.imread(os.path.join(data_path, 'im1.png'))
            img2 = cv2.imread(os.path.join(data_path, 'im2.png'))
            img3 = cv2.imread(os.path.join(data_path, 'im3.png'))

            h_len = 1
            w_len = 2

            for h in range(h_len):
                for w in range(w_len):
                    img1_ = img1[128*h : 128*h+256, 188*w:188*w+256, :]
                    img2_ = img2[128*h : 128*h+256, 188*w:188*w+256, :]
                    img3_ = img3[128*h : 128*h+256, 188*w:188*w+256, :]

                    patch_outpath = os.path.join(out_path, '{}_{}'.format(h, w))
                    os.makedirs(patch_outpath, exist_ok=True)

                    cv2.imwrite(os.path.join(patch_outpath, 'im1.png'), img1_)
                    cv2.imwrite(os.path.join(patch_outpath, 'im2.png'), img2_)
                    cv2.imwrite(os.path.join(patch_outpath, 'im3.png'), img3_)

                    patch_train_txt.write('{}\n'.format(os.path.join(trainlist[i], '{}_{}'.format(h, w))))

        elif trainlist[i].startswith("02"):
            img1 = cv2.imread(os.path.join(data_path, 'im1.png'))
            img2 = cv2.imread(os.path.join(data_path, 'im2.png'))
            img3 = cv2.imread(os.path.join(data_path, 'im3.png'))

            h_len = 3
            w_len = 5

            for h in range(h_len):
                for w in range(w_len):
                    img1_ = img1[128*h : 128*h+256, 160*w:160*w+256, :]
                    img2_ = img2[128*h : 128*h+256, 160*w:160*w+256, :]
                    img3_ = img3[128*h : 128*h+256, 160*w:160*w+256, :]

                    patch_outpath = os.path.join(out_path, '{}_{}'.format(h, w))
                    os.makedirs(patch_outpath, exist_ok=True)

                    cv2.imwrite(os.path.join(patch_outpath, 'im1.png'), img1_)
                    cv2.imwrite(os.path.join(patch_outpath, 'im2.png'), img2_)
                    cv2.imwrite(os.path.join(patch_outpath, 'im3.png'), img3_)

                    patch_train_txt.write('{}\n'.format(os.path.join(trainlist[i], '{}_{}'.format(h, w))))
        else:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../data')
    parser.add_argument('--output_path', type=str, default='../patch_data')

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    centercrop_ti(input_path, output_path)