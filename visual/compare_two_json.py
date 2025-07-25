import json
import os 
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser('compare_two_json')

    parser.add_argument('--result_root_folder', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results', help='root path of gt and pred results')
    parser.add_argument('--compare_json_name_A', type=str, default='BQNet.json', help='json path 1')
    parser.add_argument('--compare_json_name_B', type=str, default='IPDN.json', help='json path 2')

    parser.add_argument('--scanrefer_val_kv', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results/scanrefer_val_kv.json')

    parser.add_argument('--thre', type=int, default=0, help='thre')

    args = parser.parse_args()
    save_json_path = os.path.join(args.result_root_folder, args.compare_json_name_A[:-5]+'_'+args.compare_json_name_B)
    json_path1 = os.path.join(args.result_root_folder, args.compare_json_name_A)
    json_path2 = os.path.join(args.result_root_folder, args.compare_json_name_B)

    with open(json_path1, 'r') as f1:
        j1 = json.load(f1)
    with open(json_path2, 'r') as f2:
        j2 = json.load(f2)
    with open(args.scanrefer_val_kv, 'r') as f3:
        kv = json.load(f3)


    json_keys = j1.keys()
    res_jsons = {}
    for k in json_keys:
        f1_iou = j1[k]
        f2_iou = j2[k]
        description = kv[k]
        if f1_iou > f2_iou + args.thre:
            res_jsons[k] = {args.compare_json_name_A[:-5]:f1_iou, args.compare_json_name_B[:-5]:f2_iou,'description':description}
    with open(save_json_path, 'w') as f:
        json.dump(res_jsons, f, indent= 4)
    
    print('save done')