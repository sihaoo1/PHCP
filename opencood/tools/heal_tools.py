# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from pyexpat import model
import random
import numpy as np
import torch
import os
import sys
from collections import OrderedDict
import glob
import re

def get_model_path_from_dir(model_dir):
    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            raise "No checkpoint!"
        
        return os.path.join(save_dir, f'net_epoch{initial_epoch_}.pth')

    file_list = glob.glob(os.path.join(model_dir, 'net_epoch_bestval_at*.pth'))

    if len(file_list):
        assert len(file_list) == 1
        model_path = file_list[0]
    else:
        model_path = findLastCheckpoint(model_dir)

    print(f"find {model_path}.")
    
    return model_path


def rename_to_new_version(checkpoint_path):
    # stage1 model to new vesrion
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        # 将 'model.model' 替换为 'channel_align.model'
        new_key = key.replace('model.model', 'channel_align.model')
        new_key = new_key.replace('model.warpnet', 'warpnet')
        new_state_dict[new_key] = old_state_dict[key]


    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def remove_m4_trunk(checkpoint_path):
    # 加载 checkpoint
    old_state_dict = torch.load(checkpoint_path)

    # 创建一个新的字典，用于保存重命名后的键值对
    new_state_dict = OrderedDict()

    # 遍历旧的 state_dict，将所有的键进行重命名，然后保存到新的字典中
    for key in old_state_dict:
        if key.startswith("encoder_m4.camencode.trunk") or \
            key.startswith('encoder_m4.camencode.final_conv') or \
            key.startswith("encoder_m4.camencode.layer3"):
            continue

        new_state_dict[key] = old_state_dict[key]

    # 保存新的 checkpoint
    torch.save(new_state_dict, checkpoint_path)
    torch.save(old_state_dict, checkpoint_path.replace(".pth", ".pth.oldversion"))

def merge_dict(single_model_dict, stage1_model_dict):
    merged_dict = OrderedDict()
    single_keys = set(single_model_dict.keys())
    stage1_keys = set(stage1_model_dict.keys())
    symm_diff_set = single_keys & stage1_keys
    overlap_module = set([key.split(".")[0] for key in symm_diff_set])
    print("=======Overlap modules in two checkpoints=======")
    print(*overlap_module, sep="\n")
    for param in symm_diff_set:
        if not torch.equal(single_model_dict[param], stage1_model_dict[param]):
            print(f"[WARNING]: Different param in {param}")
    print("================================================")

    for key in single_model_dict:
        # remove keys like 'layers_m4.resnet.layer2.0.bn1.bias' / 'cls_head_m4.weight' / 'shrink_conv_m4.weight'
        # from single_model_dict
        if 'layers_m' in key or 'head_m' in key or 'shrink_conv_m' in key: 
            print(f"Pass {key}")
            continue
        merged_dict[key] = single_model_dict[key]

    for key in stage1_keys:
        merged_dict[key] = stage1_model_dict[key]

    return merged_dict

def merge_agents_dict(model1_dict, model2_dict):
    merged_dict = model1_dict.copy()
    model2_keys = set(model2_dict.keys())
    record = set()
    for key in model2_keys:
        if key.startswith("encoder_m") or key.startswith("backbone_m"):
            record.add(key.split('.')[0])
            merged_dict[key] = model2_dict[key]
    print("add agents: ", record)
    return merged_dict
    
    
    
    
def merge_and_save(single_model_dir, stage1_model_dir, output_model_dir):
    single_model_path = get_model_path_from_dir(single_model_dir)
    stage1_model_path = get_model_path_from_dir(stage1_model_dir)
    single_model_dict = torch.load(single_model_path, map_location='cpu')
    stage1_model_dict = torch.load(stage1_model_path, map_location='cpu')
    merged_dict = merge_dict(single_model_dict, stage1_model_dict)
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)

def merge_and_save_final(aligned_model_dir_list, output_model_dir):
    """
    aligned_model_dir_list:
        e.g. [m2_ALIGNTO_m1_model_dir, m3_ALIGNTO_m1_model_dir, m4_ALIGNTO_m1_model_dir, m1_collaboration_base_dir]

    output_model_dir:
        model_dir.
    """
    final_dict = OrderedDict()
    for aligned_model_dir in aligned_model_dir_list:
        aligned_model_path = get_model_path_from_dir(aligned_model_dir)
        model_dict = torch.load(aligned_model_path, map_location='cpu')
        final_dict = merge_dict(final_dict, model_dict)

    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(final_dict, output_model_path)

def merge_encoder_and_save_final(agent_model_list, output_model_dir):
    assert len(agent_model_list) == 2
    agent1_dict = torch.load(get_model_path_from_dir(agent_model_list[0]), map_location='cpu')
    agent2_dict = torch.load(get_model_path_from_dir(agent_model_list[1]), map_location='cpu')
    final_dict = merge_agents_dict(agent1_dict, agent2_dict)
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(final_dict, output_model_path)


def align_encoder_and_save_final(agent_model_list, output_model_dir):
    assert len(agent_model_list) == 2
    model1_dict = torch.load(get_model_path_from_dir(agent_model_list[0]), map_location='cpu')
    merged_dict = model1_dict.copy()
    record = set()
    for key in model1_dict:
        if key.startswith("encoder_m") or key.startswith("backbone_m"):
            new_key = key.replace("_m1", "_m3")
            print("add new key: ", new_key, key)
            merged_dict[new_key] = model1_dict[key]
            
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(merged_dict, output_model_path)
    

def merge_aligner(agent_model_list, output_model_dir):
    assert len(agent_model_list) == 2
    agent1_dict = torch.load(get_model_path_from_dir(agent_model_list[0]), map_location='cpu')
    agent2_dict = torch.load(get_model_path_from_dir(agent_model_list[1]), map_location='cpu')
    for key in agent2_dict:
        if key.startswith("aligner_m3"):
            print("merge aligner: ", key)
            agent1_dict[key] = agent2_dict[key]
    
    output_model_path = os.path.join(output_model_dir, 'net_epoch1.pth')
    torch.save(agent1_dict, output_model_path)


def set_seed():
    np_seed = 303
    torch_seed = 0
    random.seed(np_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)  # 如果使用多GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'rename_to_new_version':
        checkpoint_path = sys.argv[2]
        rename_to_new_version(checkpoint_path)
    elif func == 'remove_m4_trunk':
        checkpoint_path = sys.argv[2]
        remove_m4_trunk(checkpoint_path)
    elif func == 'merge':
        single_model_dir = sys.argv[2]
        stage1_model_dir = sys.argv[3]
        output_model_dir = sys.argv[4]
        merge_and_save(single_model_dir, stage1_model_dir, output_model_dir)
    elif func == 'merge_final': 
        merge_and_save_final(sys.argv[2:-1], sys.argv[-1])
    elif func == 'merge_encoder_and_save_final':
        merge_encoder_and_save_final(sys.argv[2:-1], sys.argv[-1])
    elif func == 'align_encoder_and_save_final':
        align_encoder_and_save_final(sys.argv[2:-1], sys.argv[-1])
    elif func == 'merge_aligner':
        merge_aligner(sys.argv[2:-1], sys.argv[-1])
    else:
        raise "This function not implemented"
