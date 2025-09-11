# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib
# This file generate fake label 

import argparse
import copy
import os
import pickle
from typing import OrderedDict
import importlib
from tensorboardX import SummaryWriter
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset, build_scenario_dataset
from opencood.utils import box_utils, eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict, update_hypes
import statistics
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_results = {}


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=10,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--evaluation', type=bool, default=False, help="evaluation mode, just like zero shot")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--store_model', default=False, type=bool)
    parser.add_argument('--score', type=float, default=0.5)
    parser.add_argument('--vis', action='store_true')
    opt = parser.parse_args()
    return opt


'''
结果scenario: detection result
'''
def scenario_eval(opencood_dataset, data_loader, model, opt, hypes, left_hand, scenario_name):    
    save_path = os.path.join(opt.model_dir, "evaluate", scenario_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fake_label = []
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note

    
    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            infer_result = inference_utils.inference_by_fusion_type(batch_data, model, opencood_dataset, opt.fusion_method)
            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            eval_utils.calucate_tp_fp_all_threshold(pred_box_tensor, pred_score, gt_box_tensor, result_stat)
            if opt.save_npy:
                npy_save_path = os.path.join(save_path, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,gt_box_tensor,batch_data['ego']['origin_lidar'][0],
                                                i,npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if opt.vis and  (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(save_path, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,batch_data['ego']['origin_lidar'][0],hypes['postprocess']['gt_range'],
                                    vis_save_path,method='bev',left_hand=left_hand)
            
            '''
            keep fake label
            '''
            score_threshold = opt.score
            soft_label = False
            if score_threshold < 0:
                score_threshold = 0.2
                soft_label = True
            keep_id = pred_score > score_threshold
            print("generate label: threshold", score_threshold, soft_label)
            
            pred_box_tensor = pred_box_tensor[keep_id]
            pred_score = pred_score[keep_id]
            pred_box_center = box_utils.corner_to_center_torch(pred_box_tensor, hypes['postprocess']['order'])
            processor = opencood_dataset.post_processor
            mask = np.zeros(hypes['postprocess']['max_num'])
            mask[:len(pred_box_center)] = 1
            before_mask_pred = np.zeros((hypes['postprocess']['max_num'], 7))
            before_mask_pred[:len(pred_box_center)] = pred_box_center.cpu().numpy()
            if soft_label:
                tmp = processor.generate_label(gt_box_center=before_mask_pred,
                    anchors=opencood_dataset.anchor_box,
                    mask=mask, pseudo_score=pred_score.cpu().numpy())
            else:
                tmp = processor.generate_label(gt_box_center=before_mask_pred,
                    anchors=opencood_dataset.anchor_box,
                    mask=mask)
            
            tmp["score"] = pred_score.cpu().numpy()
            fake_label.append(tmp)
            
            
        torch.cuda.empty_cache()
    save_results[scenario_name] = fake_label
    
    return eval_utils.eval_final_results(result_stat,
                                save_path, infer_info)
    
    
      
    

def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 
    hypes = yaml_utils.load_yaml(None, opt)
    # 不用跟新!!!
    # hypes = update_hypes(hypes, opt)
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")
    print('Creating Model')
    model = train_utils.create_model(hypes)
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    assert(torch.cuda.is_available())
    model.cuda()     

    # setting noise
    np.random.seed(303)
    
    # build dataset for each noise setting
    #hypes["validate_dir"] =  'dataset/OPV2V_few_shot/5_shot/train' 
    print('Dataset Building')
    
    ## only for test..
    opencood_datasets = build_scenario_dataset(hypes, visualize=True, train=False)
    # zero shot baseline evaluation
    model.eval()
    
    with open(os.path.join(opt.model_dir, 'log.txt'), 'w') as logfile:
        ap30s, ap50s, ap70s = [], [], []
        for opencood_dataset in opencood_datasets:
            scenario_name = opencood_dataset.get_scenario_name()
            print("start scenario: ", scenario_name)
            data_loader = DataLoader(opencood_dataset,batch_size=1, num_workers=0 if opt.vis else 4,
                            collate_fn=opencood_dataset.collate_batch_test,shuffle=False, pin_memory=False,drop_last=False)
            ap30, ap50, ap70 = scenario_eval(opencood_dataset, data_loader, model, opt, hypes, left_hand, scenario_name)
            ap30s.append(ap30)
            ap50s.append(ap50)
            ap70s.append(ap70)
            logfile.write(f"{scenario_name}: {ap30:.2f}, {ap50:.2f}, {ap70:.2f}\n")
        logfile.write(f"average: {np.mean(ap30s):.2f}, {np.mean(ap50s):.2f}, {np.mean(ap70s):.2f}\n")
            
    with open(os.path.join(opt.model_dir, 'fake_label.pkl'), 'wb') as f:
        pickle.dump(save_results, f)


if __name__ == '__main__':
    main()
