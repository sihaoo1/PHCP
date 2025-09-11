# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import copy
import os
import pickle
import shutil
from turtle import left
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_scenario_dataset
from opencood.tools.heal_tools import set_seed
from opencood.utils.common_utils import update_hypes
import statistics
import matplotlib
import matplotlib.pyplot as plt

plt.ioff()  # 禁用交互模式
matplotlib.use("Agg")  # 解决 Tkinter 线程问题
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["MPLBACKEND"] = "Agg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--use_gt', action = 'store_true')
    parser.add_argument('--store_model', default=True, type=bool)
    parser.add_argument('--save_vis_interval', type=int, default=10,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument('--checkpoint_dir', type=str, default="")
    parser.add_argument('--scenario', type=str, default="")
    opt = parser.parse_args()
    return opt


def train_val_scenario(train_dataset, test_dataset, model, opt, hypes, pseudo_label=None):
    scenario_name = train_dataset.get_scenario_name()
    print(f"start scenario: {scenario_name}")
    scenario_save_path = os.path.join(opt.model_dir, "scenario", scenario_name)
    if not os.path.exists(scenario_save_path):
        os.makedirs(scenario_save_path)
    fsl_params = hypes["fsl"]
    hypes["optimizer"]["lr"] = fsl_params["lr"]
    train_loader = DataLoader(train_dataset, batch_size=fsl_params["batch_size"],
                              num_workers=0 if os.environ.get('DEBUG') else 4,
                              collate_fn=train_dataset.collate_batch_train, shuffle=False, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=0 if os.environ.get('DEBUG') else 4,
                             collate_fn=test_dataset.collate_batch_test, shuffle=False, pin_memory=False, drop_last=False)
    assert(opt.model_dir)
    # if we want to train from last checkpoint.
    saved_path = opt.model_dir
    init_epoch, model = train_utils.load_saved_model(saved_path, model)
    lowest_val_epoch = init_epoch
    
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    scheduler = train_utils.WarmupMultiStepLR(optimizer, milestones=fsl_params["step_size"],
                                              gamma=fsl_params["gamma"], warmup_factor=fsl_params["warmup_factor"], 
                                              warmup_iters=fsl_params["warmup_iters"], warmup_method="linear")
    
    writer = SummaryWriter(scenario_save_path)
    print("train scenario:", scenario_name)
    supervise_single_flag = False 
    best_model = None
        
    ## train
    for epoch in range(fsl_params["epochs"]):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        model.train()
        try:
            model.model_train_init()
        except:
            print("no model_train_init")
            
        pseudo_label_idx = 0
        pseudi_label_step = fsl_params["batch_size"]
       
        
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            if pseudo_label is not None:
                target = pseudo_label[pseudo_label_idx:pseudo_label_idx + pseudi_label_step]
                target = aggreate_train_label(target)
                pseudo_label_idx += pseudi_label_step
            else:
                target = batch_data['ego']['label_dict']
                
            final_loss = criterion(ouput_dict, target)
            criterion.logging(epoch, i, len(train_loader), writer)

            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()
        
        
        

        if (epoch + 1) % fsl_params['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(test_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    # print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(),
                       os.path.join(scenario_save_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(scenario_save_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(scenario_save_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1
                
        scheduler.step(epoch)
        train_dataset.reinitialize()
    return best_model
   


def aggreate_train_label(batch_data):
    pos_equal_one = []
    neg_equal_one = []
    targets = []
    score = []
    for data in batch_data:
        pos_equal_one.append(torch.tensor(data['pos_equal_one']))
        neg_equal_one.append(torch.tensor(data['neg_equal_one']))
        score.append(torch.tensor(data['score']))
        targets.append(torch.tensor(data['targets']))
    
    
    return {'pos_equal_one': torch.stack(pos_equal_one, dim=0).to(device), 
            'neg_equal_one': torch.stack(neg_equal_one, dim=0).to(device), 
            'targets': torch.stack(targets, dim=0).to(device)}


def main():
    set_seed()
    opt = test_parser()
    # assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 
    hypes = yaml_utils.load_yaml(None, opt)
    # hypes = update_hypes(hypes, opt)
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")
    print(hypes)
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    assert(torch.cuda.is_available())
    # setting noise
    
    
    # build dataset for each noise setting
    print('Dataset Building')
    ## only for test..
    opencood_train_datasets = build_scenario_dataset(hypes, visualize=True, train=True)
    opencood_test_datasets = build_scenario_dataset(hypes, visualize=True, train=False)
    assert len(opencood_train_datasets) == len(opencood_test_datasets)
    scenario_len = len(opencood_train_datasets)
    train_with_pseudo_label = not opt.use_gt
    print(f"train with pseudo label: {train_with_pseudo_label}")
    
    pseudo_label_all_scenario = None
    # load pseudo label
    if train_with_pseudo_label:
        with open(os.path.join(opt.model_dir, "fake_label.pkl"), "rb") as f:
            pseudo_label_all_scenario = pickle.load(f)
    
    
    for i in range(scenario_len):
        train_dataset = opencood_train_datasets[i]
        test_dataset = opencood_test_datasets[i]
        # 只在当前模型生效! 记得修改sh_todo
        copy_model = copy.deepcopy(model)
        copy_model.cuda()
        # model.cuda()
        scenario_name = train_dataset.get_scenario_name()
        scenario_save_path = os.path.join(opt.model_dir, "scenario", scenario_name)
        if train_with_pseudo_label:
            pseudo_label = pseudo_label_all_scenario[scenario_name]
        else:
            pseudo_label = None
            
        best_model = train_val_scenario(train_dataset, test_dataset,copy_model, opt, hypes, pseudo_label)
        if best_model is None:
            print("failed to fidnd best model")
            continue
        from opencood.tools.few_shot_inference import eval_scenario
       
       
        eval_scenario(test_dataset, best_model, opt, hypes, left_hand, scenario_save_path, scenario_save_path)    
    



if __name__ == '__main__':
    main()
