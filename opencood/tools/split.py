import os
import shutil
import re

_PREFIX_RE = re.compile(r"[^_.]+")


def _extract_prefix(filename):
    base = os.path.basename(filename)
    match = _PREFIX_RE.match(base)
    return match.group(0) if match else base


def split_dataset(root_dir, output_dir_A, output_dir_B, k, no_first=False):
    for first_level in sorted(os.listdir(root_dir)):
        first_level_path = os.path.join(root_dir, first_level)
        if not os.path.isdir(first_level_path):
            continue

        print(f"Processing: {first_level}")
        second_level_list = sorted(os.listdir(first_level_path))
        if no_first and second_level_list:
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


            grouped_files = {}
            for file in os.listdir(second_level_path):
                src_file = os.path.join(second_level_path, file)
                if not os.path.isfile(src_file):
                    continue
                prefix = _extract_prefix(file)
                grouped_files.setdefault(prefix, []).append(file)

            prefix_list = sorted(grouped_files.keys())
            if k > len(prefix_list):
                raise ValueError(
                    f"Requested {k} groups, but only {len(prefix_list)} available in {second_level_path}"
                )
            select_prefixes = set(prefix_list[:k])
            dest_a_path = os.path.join(output_dir_A, first_level, second_level)
            dest_b_path = os.path.join(output_dir_B, first_level, second_level)
            os.makedirs(dest_a_path, exist_ok=True)
            os.makedirs(dest_b_path, exist_ok=True)
            for prefix in prefix_list:
                dst_root = dest_a_path if prefix in select_prefixes else dest_b_path
                for file in sorted(grouped_files[prefix]):
                    src_file = os.path.join(second_level_path, file)
                    dst_file = os.path.join(dst_root, file)
                    shutil.copy2(src_file, dst_file)
            



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset by scenario.")
    parser.add_argument("--data_dir", type=str, default="/home/sihao/dataset/OPV2V/test",
                        help="Root dataset path to split.")
    parser.add_argument("--out_dir", type=str, default="/home/sihao/dataset/ShotNum_no_first",
                        help="Base output directory to create <n>_shot and <n>_shot_no_first.")
    parser.add_argument("--groups", "-g", type=int, nargs="+", required=True,
                        help="Shot counts (k) to split, e.g. 5 or 1 3 7.")

    args = parser.parse_args()

    invalid_groups = [group for group in args.groups if group <= 0]
    if invalid_groups:
        parser.error("groups must be positive integers.")

    for group in sorted(set(args.groups)):
        output_A = os.path.join(args.out_dir, f"{group}_shot", "train")
        output_B = os.path.join(args.out_dir, f"{group}_shot", "test")
        output_A_no_first = os.path.join(args.out_dir, f"{group}_shot_no_first", "train")
        output_B_no_first = os.path.join(args.out_dir, f"{group}_shot_no_first", "test")

        os.makedirs(output_A, exist_ok=True)
        os.makedirs(output_B, exist_ok=True)
        os.makedirs(output_A_no_first, exist_ok=True)
        os.makedirs(output_B_no_first, exist_ok=True)

        print(f"start splitting dataset: {output_A}, {output_B}")
        split_dataset(args.data_dir, output_A, output_B, group, False)
        print(f"finished splitting dataset: {output_A}, {output_B}")

        print(f"start splitting dataset (no_first): {output_A_no_first}, {output_B_no_first}")
        split_dataset(args.data_dir, output_A_no_first, output_B_no_first, group, True)
        print(f"finished splitting dataset (no_first): {output_A_no_first}, {output_B_no_first}")

    print("All dataset splitting tasks completed!")
