# coding=utf-8
'''
  te the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs ,predictions= [],[],[]#predictions 空列表 存储每个预测值
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        #map()函数。需要传递两个参数，第一个参数就是需要引用的函数，第二个参数是一个可迭代对象，它会把需要迭代的元素一个个的传入第一个参数我们的函数中。因为我们的map会自动将数据作为参数传进去
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        predictions = [r[2] for r in res]
        return (hits, ndcgs, predictions)
    # Single thread
    for idx in xrange(len(_testRatings)):
        (hr,ndcg,prediction) = eval_one_rating(idx)#prediciton 每个预测值
        hits.append(hr)
        ndcgs.append(ndcg)
        predictions.append(prediction)
    return (hits, ndcgs,predictions)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    # Qipeng: what's the difference between `batch_size` here and `batch_size` given in program's arguments list?
    predictions = _model.predict([users, np.array(items)], 
                                 batch_size=100, verbose=0)#每一批次的预测值
    print(predictions)
    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    # Qipeng: to remove the last-visited item...
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg,predictions)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

