import argparse

import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument('gt_path', type=str, help='gt path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args

def get_gt_bboxes(gt_path, image_name):
    gt_boxes = []
    with open(gt_path) as f:
        gt_data = json.load(f)
        anno_size = len(gt_data["annotations"])
        if anno_size == 0:
            return []
        for image_info in gt_data["images"]:
            if image_info["im_name"] == image_name:
                image_id = image_info["id"]
                l = 0
                r = anno_size - 1
                while l < r:
                    mid = (l + r) // 2
                    if gt_data["annotations"][mid]["image_id"] >= image_id:
                        r = mid
                    else:
                        l = mid+1
                assert 0 <= l and l < anno_size
                if gt_data["annotations"][l]["image_id"] != image_id:
                    return []
                while l < anno_size and gt_data["annotations"][l]["image_id"] == image_id:
                    if gt_data["annotations"][l]["ignore"] == 0:
                        gt_boxes.append(gt_data["annotations"][l]["bbox"])
                    l += 1
                break
    return gt_boxes

def draw_gt_bboxes(gt_path, image_name, image_path, output_dir):
    gt_boxes = get_gt_bboxes(gt_path, image_name)
    image = cv2.imread(image_path)
    for gt_box in gt_boxes:
        x, y, w, h = gt_box
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, image_name), image)

def get_detector_bboxes(model, image_name, output_dir):
    image = cv2.imread(image_name)
    print(type(image))
    results = inference_detector(model, image)
    if isinstance(results, tuple):
        bbox_result = results[0]
    else:
        bbox_result = results
    bboxes = np.vstack(bbox_result)
    return bboxes

def show_vis_ratio_list(gt_path):
    gt_data = json.load(open(gt_path))
    vis_ratio_list = set()
    for anno in gt_data["annotations"]:
        vis_ratio_list.add(anno["vis_ratio"])
    vis_ratio_list = list(vis_ratio_list)
    vis_ratio_list.sort()
    print(vis_ratio_list)

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def run_detector_on_dataset():
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    gt_path = args.gt_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))
    print(eval_imgs)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    # prog_bar = mmcv.ProgressBar(len(eval_imgs))
    # for im in eval_imgs:
    #     detections = mock_detector(model, im, output_dir)
    #     prog_bar.update()
    print(get_gt_bboxes("datasets/CityPersons/val_gt.json", "frankfurt_000000_000576_leftImg8bit.png"))
    print(get_detector_bboxes(model, "datasets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/frankfurt/frankfurt_000000_000576_leftImg8bit.png", output_dir))

if __name__ == '__main__':
    # run_detector_on_dataset()
    img_path = "datasets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/frankfurt_000000_001236_leftImg8bit.png"
    img_name = "frankfurt_000000_001236_leftImg8bit.png"
    gt_path = "datasets/CityPersons/val_gt.json"
    draw_gt_bboxes(gt_path, img_name, img_path, "test/")
