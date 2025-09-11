'''
按照k shot 划分数据集，前k个为训练集 后面为测试集
'''

import os
import shutil
from collections import defaultdict

from cv2 import data

def split_dataset(root_dir, output_dir_A, output_dir_B, k, k_factor=2, data_type="all", no_first=False):
    for first_level in sorted(os.listdir(root_dir)):
        first_level_path = os.path.join(root_dir, first_level)
        if not os.path.isdir(first_level_path):
            continue

        print(f"Processing: {first_level}")
        second_level_list = sorted(os.listdir(first_level_path))
        if no_first:
            print("Remove first: ", second_level_list[0])
            second_level_list = second_level_list[1:]
            
        
        for second_level in second_level_list:
            second_level_path = os.path.join(first_level_path, second_level)
            if not os.path.isdir(second_level_path):
                # copy yaml file
                target_A = os.path.join(output_dir_A, first_level)
                target_B = os.path.join(output_dir_B, first_level)
                os.makedirs(target_A, exist_ok=True)
                os.makedirs(target_B, exist_ok=True)
                shutil.copy2(second_level_path, target_A)
                shutil.copy2(second_level_path, target_B)
                continue


            file_list = []
            for file in os.listdir(second_level_path):
                if data_type == "all":
                    file_list.append(file)
                elif  data_type == "heter" and file.endswith(".pcd"):
                    file_list.append(file)
                elif data_type == "camera" and file.endswith(".png"):
                    file_list.append(file)

            
            assert(len(file_list) % k_factor == 0)
            select_k = k * k_factor  
            file_list = sorted(file_list)
            dest_a_path = os.path.join(output_dir_A, first_level, second_level)
            dest_b_path = os.path.join(output_dir_B, first_level, second_level)
            os.makedirs(dest_a_path, exist_ok=True)
            os.makedirs(dest_b_path, exist_ok=True)
            for file in file_list[:select_k]:
                src_file = os.path.join(second_level_path, file)
                dst_file = os.path.join(dest_a_path, file)
                
                shutil.copy2(src_file, dst_file)
            
            for file in file_list[select_k:]:
                
                src_file = os.path.join(second_level_path, file)
                dst_file = os.path.join(dest_b_path, file)
                shutil.copy2(src_file, dst_file)
            





if __name__ == "__main__":
    # # ！！！记住替换放缩因子的k
    # root_directory = "/home/sihao/dataset/OPV2V_Hetero_few_shot/test"  # 替换为你的数据集路径
    # output_A = "/home/sihao/dataset/OPV2V_Hetero_few_shot/5_shot/train"  # 替换为 A 目录路径
    # output_B = "/home/sihao/dataset/OPV2V_Hetero_few_shot/5_shot/test"  # 替换为 B 目录路径
    # k_value = 5  # 设定前 k 组存入 A
    # # lidar pcd heter 是2g個一組 , png五個一組。
    # data_factor_dict = {"lidar": 6, "heter": 2, "camera": 5}
    # data_type = "camera"
    # k_factor = data_factor_dict[data_type]
    # split_dataset(root_directory, output_A, output_B, k_value, k_factor, data_type)
    # print("数据划分完成！")

    # 原始数据集路径
    root_directory = "/home/sihao/dataset/OPV2V/test"

    # 设置基础路径
    base_output_dir = "/home/sihao/dataset/ShotNum_no_first"

    # 设定前 k 组存入 A
    k_value = 5  

    # lidar、pcd、heter 是 2G 一组，png 5 个一组
    data_factor_dict = {"all": 6, "heter": 2, "camera": 5}
    data_type = "all"
    k_factor = data_factor_dict[data_type]

    # 创建 10 个子目录，并执行数据划分
    for i in range(1, 11):  # 生成 OPV2V_1 到 OPV2V_10
        if i == 5:
            continue
        output_A = os.path.join(base_output_dir, f"OPV2V_{i}/train")
        output_B = os.path.join(base_output_dir, f"OPV2V_{i}/test")

        # 确保目录存在
        os.makedirs(output_A, exist_ok=True)
        os.makedirs(output_B, exist_ok=True)

        # 运行数据划分函数
        print(f"开始划分数据集：{output_A}, {output_B}")
        split_dataset(root_directory, output_A, output_B, i, data_factor_dict["all"], "all", True)
        print(f"数据划分完成：{output_A}, {output_B}")

    print("所有数据集划分任务完成！")
