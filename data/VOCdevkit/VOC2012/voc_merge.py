import shutil
import argparse
import os.path as osp

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge PASCAL VOC Train and Aug annotations')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path

    out_dir = osp.join(devkit_path, 'VOC2012', 'SegmentationClassTrainAug')
    mkdir_or_exist(out_dir)

    print('Reading train split...')
    with open(osp.join(devkit_path, 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt')) as f:
        train_list = [line.strip() for line in f]

    print('Copying train split...')
    for name in train_list:
        src_path = osp.join(devkit_path, 'VOC2012', 'SegmentationClass', f'{name}.png')
        dst_path = osp.join(out_dir, f'{name}.png')
        shutil.copy2(src_path, dst_path)

    print('Reading aug split...')
    with open(osp.join(devkit_path, 'VOC2012', 'ImageSets', 'Segmentation', 'aug.txt')) as f:
        aug_list = [line.strip() for line in f]

    print('Copying aug split...')
    for name in aug_list:
        src_path = osp.join(devkit_path, 'VOC2012', 'SegmentationClassAug', f'{name}.png')
        dst_path = osp.join(out_dir, f'{name}.png')
        shutil.copy2(src_path, dst_path)

    print('Done!')


if __name__ == '__main__':
    main()
