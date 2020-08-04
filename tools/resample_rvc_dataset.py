import os
import copy
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def get_anns(image_ids, image_ann_map):
    anns = []
    for img_id in tqdm(image_ids, mininterval=1):
        anns += img_ann_map[img_id]
        
    return anns

def sub_sample_dataset(images, fraction):
    num_images = round(len(images) * .4)
    idx = np.random.choice(len(images), num_images, replace=False)
    idx.sort()
    return list(np.array(images)[idx])

def sample_dataset(images, annotations, fraction):
    times = int(fraction // 1)
    res = fraction % 1
    
    sampled_images = []
    sampled_annotations = []
    
    for i in range(times):
        for img in images:
            img_copy = copy.deepcopy(img)
            img_copy['id'] = f"{img['id']}_{i}"
            sampled_images.append(img_copy)
        for ann in annotations:
            ann_copy = copy.deepcopy(ann)
            ann_copy['id'] = f"{ann['id']}_{i}"
            ann_copy['image_id'] = f"{ann['image_id']}_{i}"
            sampled_annotations.append(ann_copy)
        
    if res:
        res_images = sub_sample_dataset(images, res)
        res_image_ids = [x['id'] for x in res_images]
        res_anns = [x for x in annotations if x['image_id'] in res_image_ids]
        
        for img in res_images:
            img_copy = copy.deepcopy(img)
            img_copy['id'] = f"{img['id']}_{times}"
            sampled_images.append(img_copy)
        for ann in res_anns:
            ann_copy = copy.deepcopy(ann)
            ann_copy['id'] = f"{ann['id']}_{times}"
            ann_copy['image_id'] = f"{ann['image_id']}_{times}"
            sampled_annotations.append(ann_copy)
            
    return sampled_images, sampled_annotations

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('joined_dataset', help='joined dataset to resample')
    parser.add_argument('--xcoco', type=float, default=1, help='COCO resampling factor')
    parser.add_argument('--xoid', type=float, default=1, help='OpenImages resampling factor')
    parser.add_argument('--xmvd', type=float, default=1, help='MVD resampling factor')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Store fraction
    fractions = {'coco': args.xcoco, 'oid': args.xoid, 'mvd': args.xmvd}
    
    # load data
    joint_train = json.load(open(args.joined_dataset))
    coco_images = [x for x in joint_train['images'] if 'coco' in x['file_name']]
    oid_images = [x for x in joint_train['images'] if 'oid' in x['file_name']]
    mvd_images = [x for x in joint_train['images'] if 'mvd' in x['file_name']]
    
    # print statistics before sampling
    print("Dataset statistics")
    all_img_perc = len(joint_train['images'])/100
    print(f"images   coco:{len(coco_images)}   oid: {len(oid_images)}   mvd: {len(mvd_images)}")
    print(f"% of images  coco:{len(coco_images)/all_img_perc:.1f}   oid: {len(oid_images)/all_img_perc:.1f}   mvd: {len(mvd_images)/all_img_perc:.1f}")

    # get img_ann_map
    img_ann_map = defaultdict(list)
    for ann in tqdm(joint_train['annotations']):
        img_ann_map[ann['image_id']].append(ann)

    
    # Sample Data
    coco_image_ids = [x['id'] for x in coco_images]
    mvd_image_ids = [x['id'] for x in mvd_images]
    oid_image_ids = [x['id'] for x in oid_images]

    coco_anns = get_anns(coco_image_ids, img_ann_map)
    mvd_anns = get_anns(mvd_image_ids, img_ann_map)
    mvd_anns = get_anns(oid_image_ids, img_ann_map)

    sampled_coco_imgs, sampled_coco_anns = sample_dataset(coco_images, coco_anns, fractions['coco'])
    sampled_mvd_imgs, sampled_mvd_anns = sample_dataset(mvd_images, mvd_anns, fractions['mvd'])
    sampled_oid_imgs, sampled_oid_anns = sample_dataset(oid_images, oid_anns, fractions['oid'])
    
    # print statistics after sampling
    print("Resampled dataset statistics")
    all_img_perc = len(joint_train['images'])/100
    print(f"images   coco:{len(sampled_coco_imgs)}   oid: {len(sampled_oid_imgs)}   mvd: {len(sampled_mvd_imgs)}")
    print(f"% of images  coco:{len(sampled_coco_imgs)/all_img_perc:.1f}   oid: {len(sampled_oid_imgs)/all_img_perc:.1f}   mvd: {len(sampled_mvd_imgs)/all_img_perc:.1f}")
    
    # save resampled dataset
    sampled_imgs = sampled_coco_imgs + sampled_oid_imgs + sampled_mvd_imgs
    sampled_anns = sampled_coco_anns + sampled_oid_anns + sampled_mvd_anns
    sampled_joint_train = {'categories': joint_train['categories'], 'images': sampled_imgs, 'annotations': sampled_anns}
    out_file = os.path.splitext(args.joined_dataset)[0] + '_sampled.json'
    json.dump(sampled_joint_train, open(out_file, 'w'))


if __name__ == '__main__':
    main()
