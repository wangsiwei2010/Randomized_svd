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
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets.samples_generator import make_low_rank_matrix

def compute_bench(samples_range, features_range, n_iter=3, rank=50):

    it = 0

    results = []
    results1 = []
    results2 = []
    
    max_it = len(samples_range) * len(features_range)
    for n_samples in samples_range:
        for n_features in features_range:
            it += 1
            print('====================')
            print('Iteration %03d of %03d' % (it, max_it))
            print('====================')
            X = make_low_rank_matrix(n_samples, n_features,
                                  effective_rank=rank,
                                  tail_strength=0.2)

            gc.collect()
            print("benchmarking scipy svd: ")
            tstart = time()
            svd(X, full_matrices=False)
            results.append(time() - tstart)

            gc.collect()
            print("benchmarking randomized_svd: n_iter=0")
            tstart = time()
            randomized_svd(X, rank, n_iter=0)
            results1.append(
                time() - tstart)

            gc.collect()
            print("benchmarking randomized_svd: n_iter=%d "
                  % n_iter)
            tstart = time()
            randomized_svd(X, rank, n_iter=n_iter)
            results2.append(time() - tstart)

    return results,results1,results2


    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    samples_range = np.array([10000])
    features_range = np.arange(1000, 5000, 500).astype(np.int)
    a,b,c = compute_bench(samples_range, features_range)
    x = features_range
    label = 'scikit-learn singular value decomposition benchmark results'
    fig = plt.figure(label)
           
        
    plt.xlabel('feature_samples')
    plt.ylabel('times')
    plt.plot(x,a,'b*')
    plt.plot(x,b,'g^')
    plt.plot(x,c,'ys')
    plt.legend(['scipy svd','randomized_svd: n_iter=0','randomized_svd: n_iter=3'])
    plt.show()

