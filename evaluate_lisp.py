# coding=utf-8
'''
Created on July 25, 2021
除了神经网络训练，其他脚本都采用python3执行
@author: Qipeng Song (qpsong@gmail.com)
'''
import pandas as pd
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    return parser.parse_args()


def dateset2dict(file_name):
    '''
        This utility is used to convert a given dataset file to a corresponding dict.
    '''
    result = defaultdict(list)
    with open(file_name, "r") as f:
        # skip 1st line
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            result[user].append(item)
            line = f.readline()
    return result

def predicte2dict(file_name):

def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value, dict1[key]]
    return dict3


if __name__ == '__main__':
    # args = parse_args()
    # dataset_path = args.path + args.dataset + ".train.rating")
    # init_cache = dataset.trainDataFrame

    prev_dataset_p =

    # init_cache_f = "./Data/auckland-8-1min.train.rating"
    predict_f = "./test.csv"


    csv_data = pd.read_csv(predict_f)

