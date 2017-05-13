# -*- coding: utf-8 -*-
"""
Created on  April 23 16:52:58 2017

@author: wangsiwei
"""

import six
import gc
from time import time
import numpy as np
from collections import defaultdict

from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd,norm
from sklearn.datasets.samples_generator import make_low_rank_matrix

def compute_bench(samples_range, features_range, iter_range=[3], rank_range=[50]):

    it = 0

    results = []
    results1 = []
    results2 = []
    
    max_it = len(samples_range) * len(features_range)*len(rank_range)*len(iter_range)
    for n_samples in samples_range:
        for n_features in features_range:
            for rank in rank_range:
                for n_iter in iter_range:
                    it += 1
                    print('====================')
                    print('Iteration %03d of %03d' % (it, max_it))
                    print('====================')
                    X = make_low_rank_matrix(n_samples, n_features,
                                          effective_rank=rank,
                                          tail_strength=0.2)
                    value = norm(X)
                    gc.collect()
                    print("benchmarking scipy svd: ")
                    tstart = time()
                    s1,v1,d1 = svd(X, full_matrices=False)
                    value1 = v1[0]
                    results.append(value1/value1)
        
                    gc.collect()
                    print("benchmarking randomized_svd: n_iter=1")
                    tstart = time()
                    s2,v2,d2=randomized_svd(X, rank, n_iter=1)
                    value2=v2[0]
                    results1.append(value2/value1)
        
                    gc.collect()
                    print("benchmarking randomized_svd: n_iter=%d "
                          % iter_range[0])
                    tstart = time()
                    s3,v3,d3=randomized_svd(X, rank, n_iter=n_iter)
                    value3 = v3[0]
                    results2.append(value3/value1)

    return results,results1,results2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    features_range= np.arange(100,1000,100).astype(np.int)
    samples_range = np.array([5000])
     
    a,b,c = compute_bench(samples_range, features_range)
    x = features_range
    label = 'scikit-learn singular value decomposition benchmark results'
    fig = plt.figure(label)
   

    plt.xlabel('differnt features')
    plt.ylabel('times')
    plt.plot(x,a,'ys')
    plt.plot(x,b,'m*')
    plt.plot(x,c,'rD')
    plt.legend(['scipy svd','randomized_svd: n_iter=1','randomized_svd: n_iter=3'])
    plt.show()

