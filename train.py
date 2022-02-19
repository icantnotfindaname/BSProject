import numpy as np
import os
import time
import csv
import pandas as pd
import kdtree
from glcm import glcm
from config import read_config
from randf import rf

train_dataset_dir = read_config('train_dataset_dir')
test_dataset_dir = read_config('test_dataset_dir')
train_label_dir = read_config('train_label_dir')
test_label_dir = read_config('test_label_dir')
train_result_dir = read_config('train_result_dir')
train_glcm_dir = read_config('train_glcm_dir')
test_result_dir = read_config('test_result_dir')
test_glcm_dir = read_config('test_glcm_dir')
img_dir = read_config('img_dir')
train_label_path = train_label_dir + '/train.csv'
test_label_path = test_label_dir + '/test.csv'
train_label_path_result = train_result_dir + '/train_label.csv'
train_glcm_path_result = train_result_dir + '/train_glcm.csv'
test_label_path_result = test_result_dir + '/test_label.csv'
test_glcm_path_result = test_result_dir + '/test_glcm.csv'

# read csv
train_label_reader = pd.read_csv(train_label_path)
test_label_reader = pd.read_csv(test_label_path)
train_label_list = np.array(train_label_reader).tolist()
test_label_list = np.array(test_label_reader).tolist()
print(train_label_list[0])


def glcm_all(type='train'):
    if type == 'train':
        print('Calculate all train data\'s glcm value......')
        evalue_list = []
        label_list = []
        count = 0
        size = len(train_label_list)
        for i in train_label_list:
            # print(i[0])
            evalue = glcm(i[0], train_dataset_dir)
            evalue_list.append(evalue)
            label_list.append(i[1])
            # print(count, end=' ')
            if count > int(size / 100):
                print('#', end='')
                count = 0
            else:
                count += 1
        print(evalue_list[0])
        evalue_list_ = pd.DataFrame(columns=['mean', 'variance', 'homogeneity', 'contrast',
                                            'dissimilarity', 'entroy', 'energy', 'correlation', 'ASM'], data=evalue_list)
        # evalue_list_ = pd.DataFrame(columns=['homogeneity', 'contrast',
        #                                      'dissimilarity', 'entroy', 'correlation', 'ASM'], data=evalue_list)
        label_list_ = pd.DataFrame(columns=['Label'], data=label_list)
        evalue_list_.to_csv(train_glcm_path_result,
                            encoding='utf-8', index=False)
        label_list_.to_csv(train_label_path_result,
                           encoding='utf-8', index=False)
    elif type == 'test':
        print('Calculate all test data\'s glcm value......')
        evalue_list = []
        label_list = []
        count = 0
        size = len(test_label_list)
        for i in test_label_list:
            # print(i[0])
            evalue = glcm(i[0], test_dataset_dir)
            evalue_list.append(evalue)
            label_list.append(i[1])
            # print(count, end=' ')
            if count > int(size / 100):
                print('#', end='')
                count = 0
            else:
                count += 1
        evalue_list_ = pd.DataFrame(columns=['mean', 'variance', 'homogeneity', 'contrast',
                                            'dissimilarity', 'entroy', 'energy', 'correlation', 'ASM'], data=evalue_list)
        # evalue_list_ = pd.DataFrame(columns=['homogeneity', 'contrast',
        #                                      'dissimilarity', 'entroy', 'correlation', 'ASM'], data=evalue_list)
        label_list_ = pd.DataFrame(columns=['Label'], data=label_list)
        evalue_list_.to_csv(test_glcm_path_result,
                            encoding='utf-8', index=False)
        label_list_.to_csv(test_label_path_result,
                           encoding='utf-8', index=False)
    else:
        print("Error!")


if __name__ == '__main__':
    # train glcm
    start = time.time()
    glcm_all()
    end = time.time()
    print("Done! Runing time: ", end - start)

    # test glcm
    start = time.time()
    glcm_all('test')
    end = time.time()
    print("Done! Runing time: ", end - start)

    # load train dataset
    train_glcm_reader = pd.read_csv(train_glcm_path_result)
    train_label_reader = pd.read_csv(train_label_path_result)
    train_glcm_list = np.array(train_glcm_reader)
    train_label_list = np.array(train_label_reader)
    print(train_glcm_list[0])

    # load test dataset
    test_glcm_reader = pd.read_csv(test_glcm_path_result)
    test_label_reader = pd.read_csv(test_label_path_result)
    test_glcm_list = np.array(test_glcm_reader)
    test_label_list = np.array(test_label_reader)

    print(train_label_reader.shape)
    print(train_label_list.data)

    # random forest
    rf(train_glcm_list, train_label_list.ravel(), test_glcm_list, test_label_list.ravel())

    # # build kd-tree
    # # for k in range(10):
    # k = 1
    # knn = kdtree.KNNClassifier(k, 2)
    # knn.fit(train_glcm_list, train_label_list)
    # predict_df = knn.predict(test_glcm_list)
    # predict_df.to_csv('./result/predict.csv', encoding='utf-8', index=False)

    # # accuarcy
    # correct_count = 0
    # predict_list = np.array(predict_df).tolist()
    # for i in range(len(test_label_list)):
    #     if (test_label_list[i][0] == int(predict_list[i][9])):
    #         correct_count += 1

    # print("correct_count: %d", correct_count)
    # print("accuarcy: %f", correct_count / len(test_label_list))
