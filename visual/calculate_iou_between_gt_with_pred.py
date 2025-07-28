import json
import os
import numpy as np
import argparse
from tqdm import tqdm
def rle_decode(rle_dict):
    """从 RLE 字典解码回二进制掩码"""
    length = rle_dict['length']
    counts = list(map(int, rle_dict['counts'].split()))
    mask = np.zeros(length, dtype=np.bool_)
    current_pos = 0
    for i in range(0, len(counts), 2):
        start = counts[i] - 1  # RLE 编码从1开始计数
        end = start + counts[i+1]
        mask[start:end] = 1
        current_pos = end
    return mask
if __name__ == '__main__':

    parser = argparse.ArgumentParser('calculate_iou_between_gt_with_pred')

    parser.add_argument('--result_root_folder', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results', help='root path of gt and pred results')
    parser.add_argument('--save_json_name', type=str, default='BQNet.json', help='json path')

    args = parser.parse_args()

    result_root_folder = args.result_root_folder
    save_json_path = os.path.join(result_root_folder, args.save_json_name)
    gt_res_folder = os.path.join(result_root_folder, 'gt_pmasks')
    pred_res_folder = os.path.join(result_root_folder, 'pred_pmasks')


    save_json = {}
    for idx, (gt_file, per_file) in tqdm(enumerate(zip(os.listdir(gt_res_folder), os.listdir(pred_res_folder)))):
        # if idx==5:break
        # print(gt_file, per_file)
        with open(os.path.join(gt_res_folder, gt_file), 'r') as f1:
            gt_file_json = json.load(f1)
        with open(os.path.join(pred_res_folder, per_file), 'r') as f2:
            pred_file_json = json.load(f2)        
        decoded_gt_mask  = rle_decode(gt_file_json)
        decoded_pred_mask  = rle_decode(pred_file_json)

        intersection = sum(decoded_gt_mask*decoded_pred_mask)
        union = sum(decoded_gt_mask) + sum(decoded_pred_mask) - intersection
        res_iou = intersection / (union + 1e-6)
        save_json[gt_file[:-5]] = res_iou
        # print(gt_file[:-5],' ', res_iou)
    with open(save_json_path,'w') as f3:
        json.dump(save_json, f3, indent=4)
        
    print('save done')

        

