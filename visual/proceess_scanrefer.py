import json
import os 
import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser('proceess_scanrefer')

    parser.add_argument('--json_path', type=str, default='/sdc1/songcl/IPDN/data/scanrefer/ScanRefer_val_new.json')
    parser.add_argument('--save_json_path', type=str, default='/sdc1/songcl/IPDN/code/IPDN/BQNet/BQNet_res/20250626_230823/results/scanrefer_val_kv.json')


    args = parser.parse_args()

    save_json = {}
    with open(args.json_path, 'r') as f1:
        j1 = json.load(f1)



    for jj in j1:
        # print(jj)
        strr = jj['scene_id']+'_['+jj['object_id']+']_'+jj['ann_id']
        description = jj['description']
        save_json[strr] = description

    with open(args.save_json_path, 'w') as f:
        json.dump(save_json, f, indent= 4)


    
    print('save done')