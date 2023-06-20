import time
import os
#import hdbscan
import statistics
import warnings
import clustbench
import sklearn.cluster
from sklearn.preprocessing import StandardScaler
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle, islice
import classix
from sklearn import metrics

def real_data_benchmark(cluster_function):
    np.random.seed(0)
    
    data_path = os.path.join(os.getcwd(), "data")

    # Define names of datasets that are 2d (can be easily visualised)
    data_vis = [('wut','cross'), ('wut', 'isolation'), ('wut', 'mk2'), 
                ('sipu', 'a3'), ('sipu', 'birch1'), ('sipu', 'unbalance'), 
                ('fcps', 'twodiamonds'), ('fcps', 'wingnut'),
                ('graves', 'dense'), ('graves', 'ring_noisy'),
                ('other', 'chameleon_t4_8k'), ('other', 'chameleon_t8_8k')]

    # Define names of datasets from UCI Machine Learning Repository
    data_uci = [('uci', 'ecoli'), ('uci', 'glass'), ('uci', 'ionosphere'), ('uci', 'sonar'), 
            ('uci', 'statlog'), ('uci', 'wdbc'), ('uci', 'wine'), ('uci', 'yeast')]
    
    dataset_names = [data_vis[i][1] for i in range(len(data_vis))]+[data_uci[i][1] for i in range(len(data_uci))]

    # Create empty dataframes that will store: runtime, adjusted rand score, adjusted mutual info and number of iterations
    df = pd.DataFrame(columns=dataset_names, index=['time', 'ar', 'ami', 'iterations'])

    # Create arrays that will store data for 2 dimensional visulisation
    data_to_vis_list = np.empty(len(data_vis), dtype=np.ndarray)
    preds_to_vis_list = np.empty(len(data_vis), dtype=np.ndarray)

    best_ar = [-float('inf')]*len(data_vis)
    time_for_best_ar = np.empty(len(data_vis))
    name_of_dataset = np.empty(len(data_vis), dtype='object')

    # Iterate through datasets that will be visualised
    for i, (battery, dataset) in enumerate(data_vis):
        times=[]
        ar=[] # Adjusted Rand Score
        ami=[] # Adjusted Mutual Infortmation
        iterations_list=[]
        print("Fitting "+cluster_function+" for "+dataset+" dataset")
        
        b = clustbench.load_dataset(battery, dataset, data_path)

        # Normalise data for easier parameter selection
        data=StandardScaler().fit_transform(b.data)
        labels=b.labels[0]
        n_clusters=int(b.n_clusters)
        
        for repeat in range(5):
            preds, time_taken, iterations = fit_algo(data, cluster_function, n_clusters)
            times.append(time_taken)
            calculated_ar = metrics.adjusted_rand_score(labels, preds)
            ar.append(calculated_ar)
            ami.append(metrics.adjusted_mutual_info_score(labels, preds))
            if iterations != 0:
                iterations_list.append(iterations)
            if calculated_ar>best_ar[i]:
                best_ar[i] = calculated_ar
                time_for_best_ar[i] = time_taken
        
        df.at['time', dataset] = statistics.median(times)
        df.at['ar', dataset] = statistics.median(ar)
        df.at['ami', dataset] = statistics.median(ami)

        # Case when algorithm does count iterations and hence value is not 0
        if len(iterations_list) != 0:
            df.at['iterations', dataset] = statistics.median(iterations_list)
        
        name_of_dataset[i] = dataset
        data_to_vis_list[i] = data
        preds_to_vis_list[i] = preds

    # Cluster data that won't be visualised
    for i, (battery, dataset) in enumerate(data_uci):
        times=[]
        ar=[] # Adjusted Rand Score
        ami=[] # Adjusted Mutual Infortmation
        iterations_list=[]

        print("Fitting "+cluster_function+" for "+dataset+" dataset")
        
        b = clustbench.load_dataset(battery, dataset, data_path)

        # Normalise data for easier parameter selection
        data=StandardScaler().fit_transform(b.data)
        labels=b.labels[0]
        n_clusters=int(b.n_clusters)

        for repeat in range(5):
            preds, time_taken, iterations = fit_algo(data, cluster_function, n_clusters)
            times.append(time_taken)
            ar.append(metrics.adjusted_rand_score(labels, preds))
            ami.append(metrics.adjusted_mutual_info_score(labels, preds))
            if iterations != 0:
                iterations_list.append(iterations)
        
        df.at['time', dataset] = statistics.median(times)
        df.at['ar', dataset] = statistics.median(ar)
        df.at['ami', dataset] = statistics.median(ami)

        # Case when algorithm does count iterations and hence value is not 0
        if len(iterations_list) != 0:
            df.at['iterations', dataset] = statistics.median(iterations_list)

    # Save plots in .png file
    plot_2d_cluster(data_to_vis_list, preds_to_vis_list, cluster_function+'/2dplot', cluster_function, 
                    best_ar, time_for_best_ar, name_of_dataset)

    return df

def fit_algo(data, cluster_function, n_clusters):
    n_iterations=0
    if cluster_function == 'kmeans':
        start_time = time.time()
        k_means = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
        k_means.fit(data)
        time_taken = time.time() - start_time
        preds = k_means.labels_
        n_iterations = k_means.n_iter_

    if cluster_function == 'dbscan':
        start_time = time.time()
        dbscan = sklearn.cluster.DBSCAN(eps=0.15, min_samples=5, n_jobs=1, algorithm='kd_tree')
        dbscan.fit(data)
        time_taken = time.time() - start_time
        preds = dbscan.labels_

    if cluster_function == 'classix':
        start_time = time.time()
        classix_model = classix.CLASSIX(radius=0.07, minPts=5, verbose=0)
        classix_model.fit(data)
        time_taken = time.time() - start_time
        preds = classix_model.labels_

    return preds, time_taken, n_iterations

def plot_2d_cluster(data, preds, path, alg_name, best_ar, time, name_of_dataset):
  plt.figure(figsize=(9 * 2 + 3, 13))
  plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.07)
  for i in range(len(data)):
    plt.subplot(3, 4, i+1)
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
    plt.title(name_of_dataset[i])
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
    plt.text(
            0.01,
            0.01,
            ("%.3f s" % time[i]),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="left",
        )
  plt.suptitle(alg_name, size=20)
  plt.show()
  plt.savefig(path)