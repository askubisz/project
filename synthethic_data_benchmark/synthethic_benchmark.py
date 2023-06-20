import time
#import hdbscan
import statistics
import warnings
import sklearn.cluster
from sklearn.preprocessing import StandardScaler
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
import classix
from sklearn import metrics



def benchmark_function(dataset_sizes=[500, 1500, 5000, 10000, 20000, 35000, 60000, 100000],
                         n_dimensions=[2, 5, 15, 30, 50, 100, 400, 1000],
                         n_clusters=10,
                         repeats=5,
                         cluster_function='kmeans'):
  np.random.seed(0)
  # Create empty dataframes that will store: runtime, adjusted rand score, adjusted mutual info and number of iterations
  df_time = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  df_ar = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  df_ami = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)
  df_iter = pd.DataFrame(columns=dataset_sizes, index=n_dimensions)

  # Create arrays that will store data for 2 dimensional visulisation
  dimension2_data=np.empty(len(dataset_sizes), dtype=np.ndarray)
  dimension2_preds=np.empty(len(dataset_sizes), dtype=np.ndarray)

  best_ar=[-float('inf')]*len(dataset_sizes)

  for dimension in n_dimensions:
    for i, size in enumerate(dataset_sizes):
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
        
        # Normalise data for easier parameter selection
        data=StandardScaler().fit_transform(data)

        if cluster_function == 'kmeans':
          start_time = time.time()
          k_means = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
          k_means.fit(data)
          time_taken = time.time() - start_time
          preds = k_means.labels_
          n_iterations.append(k_means.n_iter_)

        if cluster_function == 'dbscan':
          start_time = time.time()
          dbscan = sklearn.cluster.DBSCAN(eps=0.05, min_samples=5, n_jobs=1, algorithm='kd_tree')
          dbscan.fit(data)
          time_taken = time.time() - start_time
          preds = dbscan.labels_
        
        if cluster_function == 'classix':
          start_time = time.time()
          classix_model = classix.CLASSIX(radius=0.05, minPts=5, verbose=0)
          classix_model.fit(data)
          time_taken = time.time() - start_time
          preds = classix_model.labels_

        calculated_ar = metrics.adjusted_rand_score(labels, preds)

        if dimension == 2 and calculated_ar>best_ar[i]:
          dimension2_data[i] = data
          dimension2_preds[i] = preds
          best_ar[i] = calculated_ar

        times.append(time_taken)
        ar.append(calculated_ar)
        ami.append(metrics.adjusted_mutual_info_score(labels, preds))

        if time_taken > 60:
          break

      df_time.at[dimension, size] = statistics.median(times)
      df_ar.at[dimension, size] = statistics.median(ar)
      df_ami.at[dimension, size] = statistics.median(ami)
      
      # Case when algorithm does count iterations and hence the list is not empty
      if len(n_iterations)!=0:
        df_iter.at[dimension, size] = statistics.median(n_iterations)

    if dimension == 2:
      plot_2d_cluster(dimension2_data, dimension2_preds, cluster_function+'/2dplot', cluster_function, best_ar)

  return df_time, df_ar, df_ami, df_iter

def plot_2d_cluster(data, preds, path, alg_name, best_ar):
  plt.figure(figsize=(9 * 2 + 3, 13))
  plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.07)
  for i in range(len(data)):
    plt.subplot(2, 4, i+1)
    # Check how many colours are needed
    max_pred = int(max(preds[i]) + 1)
    # Create a list with 24 distinct colours that can repeat itself when need more
    colors = np.array(list(islice(cycle([
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5", "#8c6d31", "#393b79", "#ad494a", "#3182bd",]), max_pred,)))
    # Add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(data[i][:, 0], data[i][:, 1], s=10, color=colors[preds[i]])
    plt.title("n="+str(len(data[i])))
    plt.xticks(())
    plt.yticks(())
    plt.text(
            0.99,
            0.01,
            ("AR: %.2f" % best_ar[i]),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
  plt.suptitle(alg_name, size=20)
  plt.show()
  plt.savefig(path)