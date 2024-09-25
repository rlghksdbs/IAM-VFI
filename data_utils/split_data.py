import os
import argparse

def ti_class(input, T1, T2):
    input_path = input

    all_ti_list = []
    ti_easy_list = []
    ti_medium_list = []
    ti_hard_list = []

    ti = os.path.join(input_path, 'patch_ti.txt')

    with open(ti, 'r') as f:
        ti_list = f.read().splitlines()

    ti_range = len(ti_list)
    for i in range(0, ti_range, 1):
        clip_ti = [vimeo_ti.strip() for vimeo_ti in ti_list[i].split('\t')]

        clip_ti_ = float(clip_ti[1])
        all_ti_list.append(clip_ti[0])

        if clip_ti_ <= T1:
            ti_easy_list.append(clip_ti[0])
        elif T1 < clip_ti_ <= T2:
            ti_medium_list.append(clip_ti[0])
        else:
            ti_hard_list.append(clip_ti[0])


    return all_ti_list, ti_easy_list, ti_medium_list, ti_hard_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_ti_path', type=str, default='../patch_data')
    parser.add_argument('--output_path', type=str, default='../patch_data/train_text')
    parser.add_argument('--threshold1', type=int, default=15)
    parser.add_argument('--threshold2', type=int, default=30)
    args = parser.parse_args()

    ti_path = args.input_ti_path
    threshold1 = args.threshold1
    threshold2 = args.threshold2
    output_path = args.output_path + '/vimeo_x1x2_TI_{}_{}'.format(threshold1, threshold2)

    os.makedirs(output_path, exist_ok=True)

    ti_data = []
    ti_easy_data = []
    ti_medium_data = []
    ti_hard_data = []

    ti_txt = open(os.path.join(output_path, 'all.txt'), 'w')
    ti_easy_txt = open(os.path.join(output_path, 'easy.txt'), 'w')
    ti_medium_txt = open(os.path.join(output_path, 'medium.txt'), 'w')
    ti_hard_txt = open(os.path.join(output_path, 'hard.txt'), 'w')

    train_ti, train_easy, train_medium, train_hard = ti_class(ti_path, threshold1, threshold2)

    for i in range(len(train_ti)):
        ti_data.append(train_ti[i])

    for ie in range(len(train_easy)):
        ti_easy_data.append(train_easy[ie])

    for im in range(len(train_medium)):
        ti_medium_data.append(train_medium[im])

    for ih in range(len(train_hard)):
        ti_hard_data.append(train_hard[ih])


    for all in range(len(ti_data)):
        ti_txt.write('{}\n'.format(ti_data[all]))
    for e in range(len(ti_easy_data)):
        ti_easy_txt.write('{}\n'.format(ti_easy_data[e]))
    for m in range(len(ti_medium_data)):
        ti_medium_txt.write('{}\n'.format(ti_medium_data[m]))
    for h in range(len(ti_hard_data)):
        ti_hard_txt.write('{}\n'.format(ti_hard_data[h]))

    all_clip_len = len(train_ti)
    easy_clip_len = len(train_easy)
    medium_clip_len = len(train_medium)
    hard_clip_len = len(train_hard)

    final_all_len = len(ti_data)
    final_easy_len = len(ti_easy_data)
    final_medium_len = len(ti_medium_data)
    final_hard_len = len(ti_hard_data)

    if all_clip_len == final_all_len:
        print('split sucess\t all : {}'.format(all_clip_len))
    else:
        print('split miss')
    if easy_clip_len == final_easy_len:
        print('split success\t easy : {}'.format(easy_clip_len))
    else:
        print('split miss')
    if medium_clip_len == final_medium_len:
        print('split success\t medium : {}'.format(medium_clip_len))
    else:
        print('split miss')
    if hard_clip_len == final_hard_len:
        print('split success\t hard : {}'.format(hard_clip_len))
    else:
        print('split miss')