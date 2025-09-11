import argparse
import os
from re import T, sub
import shutil
import subprocess
import numpy as np


def merge_pth_and_run(source_dir, target_dir, model_run_config, vis=True):
    source_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    for folder in source_folders:
        source_path = os.path.join(source_dir, folder)
        target_path = os.path.join(target_dir, folder)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        cmd = ["python", "opencood/tools/heal_tools.py", "merge_aligner", base_dir, source_path, target_path]
        print(f"Executing: {' '.join(cmd)}")
        cmd2 = ["python", "opencood/tools/few_shot_inference.py", "--model_dir", target_path, "--scenario", folder]
        if vis:
            cmd2.append("--vis")
        
        shutil.copy(model_run_config, os.path.join(target_path, "config.yaml"))
        print(f"Executing: {' '.join(cmd2)}")
        subprocess.run(cmd, check=True)
        subprocess.run(cmd2, check=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo script for base_dir argument")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="opencood/logs/opv2v/paper/my_lidar_confidence_0.3",
        help="Base directory path"
    )
    parser.add_argument("--vis", action='store_true')
    return parser.parse_args()

def collect_logs(source_dir, output_file):
    source_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    all_values = []
    log_entries = []
    
    for subdir in source_folders:
        subdir_path = os.path.join(source_dir, subdir)
        log_file = os.path.join(subdir_path, "log.txt")
        
        if os.path.isdir(subdir_path) and os.path.isfile(log_file):
            result = extract_values_from_log(log_file)
            if result:
                timestamp, values = result
                log_entries.append(f"{timestamp}: {', '.join(map(str, values))}")
                all_values.append(values)

    if not all_values:
        print("No valid log.txt files found.")
        return
    all_values = np.array(all_values)  
    mean_values = np.round(np.mean(all_values, axis=0), 3)  

    with open(output_file, "w") as f:
        f.write("\n".join(log_entries))  
        f.write("\n")
        f.write(f"average: {', '.join(map(str, mean_values))}\n")

    print(f"Results saved in {output_file}")

def extract_values_from_log(file_path):
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()  
            parts = first_line.split(":")  
            if len(parts) < 2:
                return None  
            
            values = [float(x.strip()) for x in parts[1].split(",")]  
            if len(values) != 3:
                return None  
            return parts[0], values  
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    

if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    print(base_dir)
    base_dir = args.base_dir
    source_dir = os.path.join(base_dir, "fsl_train/scenario")
    target_dir = os.path.join(base_dir, "final")
    model_run_config = os.path.join(base_dir, "config.yaml")
    merge_pth_and_run(source_dir, target_dir, model_run_config, args.vis)
    
    output_file = os.path.join(base_dir, "log.txt")
    collect_logs(target_dir, output_file)