import numpy as np
import os
import array
import imageio
import time
import pandas as pd
from config import read_config
from numpy.lib.function_base import piecewise
from calcu_glcm import calcu_glcm, calcu_evalue

# file_name = 'D:/010 Working Station/bs/code/0A32eTdBKayjCWhZqDOQ.bytes'
# graph_name = 'D:/010 Working Station/bs/code/test.png'
# f = open(file_name, 'rb')
# ln = os.path.getsize(file_name)
# width = 2048
# rem = ln % width
# a = array.array("B")  # unit 8 array
# a.fromfile(f, ln - rem)
# f.close()
# height = int(len(a) / width)
# g = np.reshape(a, (height, width))
# g = np.uint8(g)  # Todo: normalization
# imageio.imwrite(graph_name, g)

# train_dataset_dir = read_config('train_dataset_dir')
# test_dataset_dir = read_config('test_dataset_dir')
# train_label_dir = read_config('train_label_dir')
# test_label_dir = read_config('test_label_dir')
# img_dir = read_config('img_dir')
# train_label_path = os.path.join(train_label_dir, 'train.csv')
# test_label_path = os.path.join(test_dataset_dir, 'test.csv')
# train_data_files = os.listdir(train_dataset_dir)
# test_data_files = os.listdir(test_dataset_dir)

# start timing ......
start_time = time.time()

print('Parameter Setting......')
nbits = read_config('nbits')
min_gray_value, max_gray_value = 0, nbits - 1
angles = read_config('angles')
distances = read_config('distances')


def choose_width(len):
    width = 0
    if len < 10000:
        width = 32
    elif len < 30000:
        width = 64
    elif len < 60000:
        width = 128
    elif len < 100000:
        width = 256
    elif len < 200000:
        width = 384
    elif len < 500000:
        width = 512
    elif len < 1000000:
        width = 768
    else:
        width = 1024
    return width


# print('Calculate GLCM......')
# evalue_list = []
# evalue_list_regu = []
# count = 0
# for file in train_data_files:
#     if not os.path.isdir(file):
#         file_name = os.path.join(train_dataset_dir, file)
#         print(file_name)
#         f = open(file_name, 'rb')
#         ln = os.path.getsize(file_name)
#         width = choose_width(ln)
#         rem = ln % width
#         a = array.array("B")  # unit 8 array
#         a.fromfile(f, ln - rem)
#         f.close()
#         height = int(len(a) / width)
#         img = np.reshape(a, (height, width))

#         # regularlization
#         img = np.uint8(255.0 * (img - np.min(img)) /
#                        (np.max(img) - np.min(img)))
#         # img = np.uint8(img)

#         if count < read_config('num_of_example_img'):
#             img_name = os.path.join(img_dir, file + '.png')
#             imageio.imwrite(img_name, img)

#         h, w = img.shape
#         # glcms = np.array((nbits, nbits, len(distances),
#         #               len(angles)), dtype=np.uint8)
#         glcms = calcu_glcm(img, min_gray_value, max_gray_value,
#                            nbits, distances, angles)
#         evalue = calcu_evalue(glcms)

#         # regularlization
#         mean = np.mean(evalue, axis=0)
#         sigma = np.std(evalue, axis=0)
#         evalue_regu = (evalue - mean) / sigma / 3
#         evalue_list.append(evalue)
#         evalue_list_regu.append(evalue_regu)
#         count += 1

# end_time = time.time()
# print('Running Time: ', end_time - start_time)


def glcm(file_id, dir):
    file_name = dir + '/' + str(file_id) + '.asm'
    # print(file_name)
    f = open(file_name, 'rb')
    ln = os.path.getsize(file_name)
    # ln = 14400
    width = choose_width(ln)
    # width = 120
    rem = ln % width
    a = array.array("B")  # unit 8 array
    a.fromfile(f, ln - rem)
    f.close()
    height = int(len(a) / width)
    img = np.reshape(a, (height, width))

    # regularlization
    img = np.uint8((nbits - 1) * (img - np.min(img)) /
                    (np.max(img) - np.min(img)))
    # img = np.uint8(img)

    # if count < read_config('num_of_example_img'):
    #     img_name = os.path.join(img_dir, file + '.png')
    #     imageio.imwrite(img_name, img)

    # h, w = img.shape
    # glcms = np.array((nbits, nbits, len(distances),
    #               len(angles)), dtype=np.uint8)
    glcms = calcu_glcm(img, min_gray_value, max_gray_value,
                        nbits, distances, angles)
    evalue = calcu_evalue(glcms)

    # regularlization
    mean = np.mean(evalue, axis=0)
    sigma = np.std(evalue, axis=0)
    evalue_regu = (evalue - mean) / sigma / 3
    return evalue_regu



