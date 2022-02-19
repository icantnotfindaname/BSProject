import csv
from operator import index
import numpy as np
import pandas as pd
import os
from config import read_config

# config
csv_path = read_config('original_label_path')
temp_label_dir = read_config('temp_label_dir')
train_label_dir = read_config('train_label_dir')
test_label_dir = read_config('test_label_dir')
train_label_path = train_label_dir + '/train.csv'
test_label_path = test_label_dir + 'test.csv'

TRAIN_RATIO = read_config('train_ratio')
TEST_RATIO = 1 - TRAIN_RATIO
NUM = read_config('num_of_data_per_type')
TYPE_NUM = read_config('num_of_type')
num_of_train_data_per_type = int(NUM * TRAIN_RATIO)
num_of_test_data_per_type = int(NUM * TEST_RATIO)

def create_csv(path):
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        head = ['ID', 'Class']
        csv_write.writerow(head)


# csv_reader = csv.reader(open(csv_path))
# ID, Class
# print(csv_reader.head)
# print(csv_reader.tail)
# print(l.shape)
csv_reader = pd.read_csv(csv_path)
print(csv_reader.columns)
l = np.array(csv_reader).tolist()
type = 1
train_data = np.empty((NUM * TYPE_NUM, 2))
count = 0

# divide label csv
for n in range(1, TYPE_NUM + 1):
    llist = []
    for item in l:
        if item[1] == n:
            llist.append(item)
    llist_ = pd.DataFrame(columns=['ID', 'Class'], data=llist)
    label_csv_path = './label/label_' + str(n) + '.csv'
    llist_.to_csv(label_csv_path, encoding='utf-8', index=False)

# load train data & test data
print('num_of_train_data_per_type: ', num_of_train_data_per_type)
print('num_of_test_data_per_type', num_of_test_data_per_type)
train_label_list = []
test_label_list = []
for n in range(1, TYPE_NUM + 1):
    # create_csv(train_label_path)
    # create_csv(test_label_path)
    temp_file_name = 'label_' + str(n) + '.csv'
    temp_label_path = os.path.join(temp_label_dir, temp_file_name)
    reader = pd.read_csv(temp_label_path)
    l = np.array(reader).tolist()
    if len(l) >= NUM:
        train_label_list += l[0: num_of_train_data_per_type]
        test_label_list += l[num_of_train_data_per_type: num_of_train_data_per_type + num_of_test_data_per_type]
    else: 
        temp_train_num = int(len(l) * TRAIN_RATIO)
        train_label_list += l[0: temp_train_num]
        test_label_list += l[temp_train_num: len(l)]

train_label_list_ = pd.DataFrame(columns=['ID', 'Class'], data=train_label_list)
test_label_list_ = pd.DataFrame(columns=['ID', 'Class'], data=test_label_list)
train_label_list_.to_csv(train_label_path, encoding='utf-8', index=False)
test_label_list_.to_csv(test_label_path, encoding='utf-8', index=False)


