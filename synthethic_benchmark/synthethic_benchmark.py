import time
#import hdbscan
import statistics
import warnings
import sklearn.cluster
import scipy.cluster
import sklearn.datasets
import numpy as np
import pandas as pd
from numpy.linalg import norm
#from classix.aggregation_test import aggregate
from classix import CLASSIX
#from quickshift.QuickshiftPP import *
from sklearn import metrics



def benchmark_function(dataset_sizes=[500, 1500, 5000, 10000, 20000, 35000, 60000, 100000],
                         n_dimensions=[2, 5, 15, 30, 50, 100, 400, 1000],
                         n_clusters=10,
                         repeats=5,
                         cluster_function='kmeans'):
  np.random.seed(0)
  df_time = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  df_ar = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  df_ami = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  df_iter = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  for dimension in n_dimensions:
    for size in dataset_sizes:
      times=[]
      ar=[] # Adjusted Rand Score
      ami=[] # Adjusted Mutual Infortmation
      n_iterations=[]
      print("Fitting "+cluster_function+" for "+str(size)+" data points in "+str(dimension)+" dimensions")
      for iteration in range(repeats):
        data, labels = sklearn.datasets.make_blobs(n_samples=size,
                                                   n_features=dimension,
                                                   centers=n_clusters,
                                                   cluster_std=1)
        if cluster_function == 'kmeans':
          start_time = time.time()
          k_means = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
          k_means.fit(data)
          time_taken = time.time() - start_time
          preds = k_means.labels_
          n_iterations.append(k_means.n_iter_)

        if cluster_function == 'dbscan':
          start_time = time.time()
          dbscan = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, n_jobs=1)
          dbscan.fit(data)
          time_taken = time.time() - start_time
          preds = dbscan.labels_
        
        if cluster_function == 'classix':
          start_time = time.time()
          classix = CLASSIX(radius=0.3, minPts=5, verbose=0)
          classix.fit(data)
          time_taken = time.time() - start_time
          preds = classix.labels_


        times.append(time_taken)
        ar.append(metrics.adjusted_rand_score(labels, preds))
        ami.append(metrics.adjusted_mutual_info_score(labels, preds))

        if time_taken > 60:
          break

      df_time.at[dimension, size] = statistics.median(times)
      df_ar.at[dimension, size] = statistics.median(ar)
      df_ami.at[dimension, size] = statistics.median(ami)
      if len(n_iterations)!=0:
        df_iter.at[dimension, size] = statistics.median(n_iterations)

  return df_time, df_ar, df_ami, df_iter