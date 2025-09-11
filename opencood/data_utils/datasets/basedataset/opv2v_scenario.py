from json import load
from token import OP
from torch.utils.data import Dataset
import glob
import os
from opencood.data_utils.datasets import OPV2VBaseDataset
from opencood.hypes_yaml.yaml_utils import load_yaml


m1_scenario_mask = ['2021_08_16_22_26_54', '2021_08_18_19_11_02', '2021_08_18_19_48_05', '2021_08_18_21_38_28', '2021_08_18_22_16_12', '2021_08_18_23_23_19', '2021_08_20_16_20_46', '2021_08_20_20_39_00', '2021_08_20_21_10_24', '2021_08_20_21_48_35', '2021_08_21_09_09_41', '2021_08_21_09_28_12', '2021_08_21_16_08_42', '2021_08_21_17_00_32', '2021_08_21_17_30_41', '2021_08_21_22_21_37', '2021_08_22_06_43_37', '2021_08_22_07_24_12', '2021_08_22_07_52_02', '2021_08_22_09_08_29', '2021_08_22_09_43_53', '2021_08_22_10_10_40', '2021_08_22_10_46_58', '2021_08_22_11_29_38', '2021_08_22_21_41_24', '2021_08_22_22_30_58', '2021_08_23_10_51_24', '2021_08_23_11_06_41', '2021_08_23_11_22_46', '2021_08_23_12_13_48', '2021_08_23_12_58_19', '2021_08_23_13_10_47', '2021_08_23_13_17_21', '2021_08_23_15_19_19', '2021_08_23_16_06_26', '2021_08_23_17_07_55', '2021_08_23_19_27_57', '2021_08_23_19_42_07', '2021_08_23_20_47_11', '2021_08_23_21_47_19', '2021_08_23_22_31_01', '2021_08_24_07_45_41', '2021_08_24_12_19_30', '2021_08_24_20_09_18', '2021_08_24_21_29_28', '2021_09_09_19_27_35', '2021_09_09_22_21_11', '2021_09_09_23_21_21', '2021_09_10_12_07_11', '2021_09_11_00_33_16']


class OPV2VScenarioBaseDataset():
    def __init__(self, params, visulize, train=True, select_scenario=None):
        if train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        scenario_folders = sorted([os.path.join(root_dir, x)
                                    for x in os.listdir(root_dir) if
                                    os.path.isdir(os.path.join(root_dir, x))])
        scenario_folders = [x for x in scenario_folders if '2021_09_09_13_20_58' not in x]

        # unmask non-m1 scenarios
        if 'm1_mask' in params:
            tmp = []
            for scenario in scenario_folders:
                scenario_name = scenario.split("/")[-1]
                if scenario_name not in m1_scenario_mask:
                    print("skip:", scenario)
                else:
                    tmp.append(scenario)
            scenario_folders = tmp

        self.scenarios = scenario_folders
        self.scenario_dataset = [OPV2VBaseDataset(params, visulize, train, select_scenario=x) for x in scenario_folders]
        
    def get_scenarios_dataset(self):
        return self.scenario_dataset
    
    def get_scenarios(self):
        return self.scenarios
        
if __name__ == "__main__":
    params = load_yaml("/home/sihao/repo/HEAL/opencood/hypes_yaml/opv2v/MoreModality/HEAL/stage1/m1_pyramid.yaml")
    params["root_dir"] =  "/home/sihao/dataset/OPV2V/test"
    params["m1_mask"] = 1
    opv2v_scenario = OPV2VScenarioBaseDataset(params, visulize=False, train=True)
    datasets = opv2v_scenario.get_scenarios_dataset()
    for dataset in datasets:
        print(dataset.scenario_folders)
