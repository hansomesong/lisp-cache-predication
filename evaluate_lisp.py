# coding=utf-8
'''
Created on July 25, 2021
除了神经网络训练，其他脚本都采用python3执行

我试了下拿auckland-8-6-1mins.train.rating和auckland-8-6-2mins.train.rating这两个数据集，去评估预测效率。
推送TOP10的可能被访问的item, 令HR1，HR2分别表示不采用预测和采用预测情况下的Hit Ratio，最后发现
HR1 = 0.1528, HR2 = 0.6470, 增益达到了323%！！！
刚一看到这个结果，第一反应是程序有问题，但仔细看了下数据，我觉得原因是：
1. 刚开始cache内容不多，TOP10的推送其实甚至比大部分xTR的缓存都长
2. 这个时候HR，还是爬升阶段，还没有到达稳态值，HR1会很低。
@author: Qipeng Song (qpsong@gmail.com)
'''
import pandas as pd
from collections import defaultdict
import argparse
import os
import platform
import numpy as np

# 方便DEBUG
import logging


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
        # skip 1st line, if it starts with `#`
        line = f.readline()
        if line.startswith("#"):
            line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            result[user].append(item)
            line = f.readline()
    return result

def predicte2dict(file_name):
    # create dataframe object from CSV file and regard the first column as index for this dataframe.
    # Attention:
    df = pd.read_csv(file_name, index_col=0)
    # convert type of column names to int, cause key in target dict should be int type.
    df.columns = df.columns.astype(int)
    return df.to_dict(orient='list')

def merge_dict(dict1, dict2):
    '''
    Merge dictionaries and keep values of common keys in list
    '''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            # in our case, value is a list, we need to extend it with dict1[key], which is another list.
            dict3[key] = value + dict1[key]
    return dict3

def evaluation(dataset1, dataset2):
    """
    接受两个train dataset作为输入，calculate hit ratio with/without deep-learning empowered cache prediction
    @prev, 前一个周期内形成的dataset
    @curr, 当前周期内形成的dataset
    """
    curr_cache = dateset2dict(dataset1)
    prev_test_dict = dateset2dict(dataset1.split(".")[0] + ".test.rating")
    # test for correct test file parsing:
    test_result = True
    for key, value in prev_test_dict.items():
        if len(value) != 1:
            test_result = False
    assert(test_result == True)

    query_records = dateset2dict(dataset2)

    # 别忘了把测试集的内容添加进当前时间周期内的交互历史，即query_records
    # 因为测试集中记录的是每个user最后一次访问的item, 其对应的训练集中也确实不含有该记录
    query_records = merge_dict(query_records, prev_test_dict)

    # =================test
    # query_records = query_records
    # curr_cache = merge_dict(curr_cache, prev_test_dict)
    # =================

    predict_f = os.path.join("Predication", dataset1.split(".")[0].split("/")[1] + ".predict.csv")
    predict_cache = predicte2dict(predict_f)
    # test for correct predication parsing:
    test_result = True
    for key, value in predict_cache.items():
        if len(value) != 10:
            test_result = False
    assert (test_result == True)

    cache_with_predict = merge_dict(curr_cache, predict_cache)

    # test for cache with predicated items:
    test_result = True
    for key, value in predict_cache.items():
        for item in value:
            if item not in cache_with_predict[key]:
                test_result = False
    assert(test_result==True)

    # test: after dict merge, dicts: prev_test_dict do not change:
    test_result = True
    for key, value in prev_test_dict.items():
        if len(value) != 1:
            test_result = False
    assert (test_result == True)

    avg_hr1, avg_hr2 = hiration_calculation(curr_cache, cache_with_predict, query_records)
    return avg_hr1, avg_hr2

def new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f):
    """
    接受两个train dataset作为输入，calculate hit ratio with/without deep-learning empowered cache prediction
    @accumu_cache, 截止至当前时期，累积的cache字典，函数执行中就地更新其内容
    @curr_dataset, 当前周期内交互历史字典
    @prev_test, 上个周期训练中的测试文件
    @predict_f, 根据当前周期交互历史，预测的缓存，其名称可由curr_dataset衍生而来，其内容当在当前周期评估接受后，融合进predict_accumu_cache。
    以后考虑弄个wrapper文件，自动生成上述文件名
    """
    if not accumu_cache and not predict_accumu_cache:
        # case when accumulated cache and predicated accumulated cache both are empty.
        # 但仍然需要获取当前周期内产生的交互历史，并由此更新accumu_cache
        accumu_cache = dateset2dict(curr_dataset)
        # 第一周期结束后，预测机制赋能的缓存也应更新
        # predict_f = os.path.join("Predication", curr_dataset.split(".")[0].split("/")[1] + ".predict.csv")
        predict_cache = predicte2dict(predict_f)
        predict_accumu_cache = merge_dict(accumu_cache, predict_cache)
        # print("I'm here")
        # print(predict_accumu_cache)
        # 有一点困惑：为啥accumu_cache, predict_accumu_cache在程序结束之后不会更新呢？？？
        return accumu_cache, predict_accumu_cache, 0, 0
    else:
        # 通过处理数据集文件名，获取测试集、预测缓存文件名
        query_records = dateset2dict(curr_dataset)
        # 别忘了把测试集的内容添加进当前时间周期内的交互历史，即query_records
        # 因为测试集中记录的是每个user最后一次访问的item, 其对应的训练集中也确实不含有该记录
        # "Data/auckland-8-6-1mins.test.rating"
        prev_test_dict = dateset2dict(prev_test_f)
        query_records = merge_dict(query_records, prev_test_dict)

        # 评估 Hit ratio
        avg_hr1, avg_hr2 = hiration_calculation(accumu_cache, predict_accumu_cache, query_records)
        # 最后：更新accumu_cache, predict_accumu_cache
        # predict_f 指代的是由当前训练集：curr_dataset，训练并预测的缓存
        accumu_cache = merge_dict(accumu_cache, query_records)
        predict_cache = predicte2dict(predict_f)
        predict_accumu_cache = merge_dict(predict_accumu_cache, query_records)
        predict_accumu_cache = merge_dict(predict_accumu_cache, predict_cache)
        # 返回结果
        # print("Average HR1: {}, Average HR2: {}, Gain: {:.2f}%".format(
        #     avg_hr1, avg_hr2,np.abs(avg_hr1 - avg_hr2) / avg_hr1 * 100)
        # )

        return accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2


def hiration_calculation(accumu_cache, predict_accumu_cache, query_records):
    """
    calculate hit ratio with/without predicated cache
    @curr_cache, a dict recording cache for all users at the beginning of current period.
    @query_records, a dict recording queries history for all users with current period.
    @cache_with_predict, a dict recording cache with prediction for all users at the beginning of current period.
    """
    base_hr1 = []
    base_hr2 = []
    # for key1, value1 in curr_cache.items():
    #     print("for user: {}".format(key1))
    #     if key1 in query_records:
    #         print("Previous cache: {}".format(value1))
    #         print("Current cache: {}".format(query_records[key1]))
    #         print("Previous cache with prediction: {}".format(cache_with_predict[key1]))
    #
    #         inter_section1 = list(set(value1).intersection(set(query_records[key1])))
    #         inter_section2 = list(set(cache_with_predict[key1]).intersection(set(query_records[key1])))
    #
    #         print("case1 Hit cache list: {}".format(inter_section1))
    #         print("case2 Hit cache list: {}".format(inter_section2))
    #
    #         hr1 = len(inter_section1) / len(query_records[key1])
    #         hr2 = len(inter_section2) / len(query_records[key1])
    #
    #         print("hit ratio1: {}, hit ratio2: {}".format(hr1, hr2))
    #         base_hr1.append(hr1)
    #         base_hr2.append(hr2)
    # avg_hr1, avg_hr2 = np.mean(base_hr1), np.mean(base_hr2)
    # print("Average HR1: {}, Average HR2: {}, Gain: {:.2f}%".format(avg_hr1, avg_hr2, np.abs(avg_hr1-avg_hr2)/avg_hr1*100))

    for key1, value1 in query_records.items():
        # Iterate query records
        print("for user: {}".format(key1))
        if key1 not in accumu_cache:
            # 当前周期内容，a new user 开始了询问，当前cache并无任何记录，因此直接讲hit ratio记为0
            # base_hr1.append(0)
            # base_hr2.append(0)
            pass
        elif key1 in accumu_cache:
            cache = accumu_cache[key1]
            print("current cache: {}".format(cache))
            print("query record: {}".format(value1))
            print("cache with prediction: {}".format(predict_accumu_cache[key1]))

            inter_section1 = list(set(value1).intersection(set(cache)))
            inter_section2 = list(set(value1).intersection(set(predict_accumu_cache[key1])))

            print("case1 Hit cache list: {}".format(inter_section1))
            print("case2 Hit cache list: {}".format(inter_section2))

            hr1 = len(inter_section1) / len(value1)
            hr2 = len(inter_section2) / len(value1)

            print("hit ratio1: {}, hit ratio2: {}".format(hr1, hr2))
            base_hr1.append(hr1)
            base_hr2.append(hr2)
    # maybe we need to consider weighted average in future?
    avg_hr1, avg_hr2 = np.mean(base_hr1), np.mean(base_hr2)
    print("Average HR1: {}, Average HR2: {}, Gain: {:.2f}%".format(avg_hr1, avg_hr2, np.abs(avg_hr1-avg_hr2)/avg_hr1*100))
    return avg_hr1, avg_hr2


if __name__ == '__main__':
    # args = parse_args()
    # dataset_path = args.path + args.dataset + ".train.rating")
    # init_cache = dataset.trainDataFrame
    SYSTEM=platform.system()
    DATASET_NAME = "auckland-8"
    SUB_DATASET = "6mins-traces" # numerical value refers to the duration of PCAP file
    RAWDATA_ROOT = ""

    if SYSTEM.startswith('Darwin'):
        DATASET_ROOT = os.path.join("/Users/qsong/Documents/xidian/song/research/ncf-project", DATASET_NAME, SUB_DATASET)
        # sys.exit("Raw data is stored at remote server, please start a SSH session then run this program!")

    elif SYSTEM.startswith("Linux"):
        RAWDATA_ROOT = os.path.join("/home/qsong/Documents/xidian/NCF/lisp-cache-predication/Data_Raw", DATASET_NAME, SUB_DATASET)

    prev_train_dataset_p = "Data/auckland-8-6-1mins.train.rating"
    prev_test_dataset_p = "Data/auckland-8-6-1mins.test.rating"
    curr_train_dataset_p = "Data/auckland-8-6-2mins.train.rating"

    # evaluation(prev_train_dataset_p, curr_train_dataset_p)

    accumu_cache, predict_accumu_cache = {}, {}
    avg_hr1, avg_hr2 = 0, 0
    # Period 1
    curr_dataset = "Data/auckland-8-6-1mins.train.rating"
    prev_test_f = "Data/auckland-8-6-1mins.test.rating"
    predict_f = "Predication/auckland-8-6-1mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)

    # Period 2
    curr_dataset = "Data/auckland-8-6-2mins.train.rating"
    prev_test_f = "Data/auckland-8-6-1mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-2mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)

    # Period 3
    curr_dataset = "Data/auckland-8-6-3mins.train.rating"
    prev_test_f = "Data/auckland-8-6-2mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-3mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)
    #
    # Period 4
    curr_dataset = "Data/auckland-8-6-4mins.train.rating"
    prev_test_f = "Data/auckland-8-6-3mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-4mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)
    #
    # Period 5
    curr_dataset = "Data/auckland-8-6-5mins.train.rating"
    prev_test_f = "Data/auckland-8-6-4mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-5mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)
    #
    # Period 6
    curr_dataset = "Data/auckland-8-6-6mins.train.rating"
    prev_test_f = "Data/auckland-8-6-5mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-6mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)

    # Period 7
    curr_dataset = "Data/auckland-8-6-7mins.train.rating"
    prev_test_f = "Data/auckland-8-6-6mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-7mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)

    # Period 8
    curr_dataset = "Data/auckland-8-6-8mins.train.rating"
    prev_test_f = "Data/auckland-8-6-7mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-8mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)

    # Period 9
    curr_dataset = "Data/auckland-8-6-9mins.train.rating"
    prev_test_f = "Data/auckland-8-6-8mins.test.rating" #一定是上一个周期形成的测试数据集
    predict_f = "Predication/auckland-8-6-9mins.predict.csv"
    accumu_cache, predict_accumu_cache, avg_hr1, avg_hr2 = new_evaluation(accumu_cache, predict_accumu_cache, curr_dataset, prev_test_f, predict_f)