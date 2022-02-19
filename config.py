from collections import defaultdict, OrderedDict
import json
import numpy as np

config_dict = {
    'version': '1.0',
    'save_json_path': './config.json',
    'original_label_path': './label/trainLabels.csv',
    'train_dataset_dir': './dataset/train',
    'test_dataset_dir': './dataset/train',
    'img_dir': './img/',
    'train_label_dir': './label',
    'test_label_dir': './label',
    'temp_label_dir': './label',
    'train_result_dir': './result',
    'train_glcm_dir': './result',
    'test_result_dir': './result',
    'test_glcm_dir': './result',
    'num_of_type': 9,
    'num_of_data_per_type': 10000,
    'train_ratio': 0.7,

    # glcm
    'nbits': 128,
    'angles': [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
    'distances': [1, 2, 3, 4],
    'num_of_example_img': 10,
}

def read_config(name):
    return config_dict[name]


def to_json(config_dict):
    json_str = json.dumps(config_dict, indent=4)
    with open(read_config('save_json_path'), 'w') as json_file:
        json_file.write(json_str)


if __name__ == '__main__': 
    to_json(config_dict)
    # print(type(read_config('angles')))