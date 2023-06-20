import os
import clustbench
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice

def plot_true_labels():
    np.random.seed(0)
    
    data_path = os.path.join(os.getcwd(), "data")

    # Define names of datasets that are 2d (can be easily visualised)
    data_vis = [('wut','cross'), ('wut', 'isolation'), ('wut', 'mk2'), 
                ('sipu', 'a3'), ('sipu', 'birch1'), ('sipu', 'unbalance'), 
                ('fcps', 'twodiamonds'), ('fcps', 'wingnut'),
                ('graves', 'dense'), ('graves', 'ring_noisy'),
                ('other', 'chameleon_t4_8k'), ('other', 'chameleon_t8_8k')]

    dataset_names = [data_vis[i][1] for i in range(len(data_vis))]

    # Create arrays that will store data for 2 dimensional visulisation
    data_to_vis_list = np.empty(len(data_vis), dtype=np.ndarray)
    labels_to_vis_list = np.empty(len(data_vis), dtype=np.ndarray)

    # Iterate through datasets that will be visualised
    for i, (battery, dataset) in enumerate(data_vis):
        b = clustbench.load_dataset(battery, dataset, data_path)

        # Normalise data for easier parameter selection
        data=StandardScaler().fit_transform(b.data)
        labels=b.labels[0]

        data_to_vis_list[i] = data
        labels_to_vis_list[i] = labels

    # Save plots in .png file
    plot_2d_cluster(data_to_vis_list, labels_to_vis_list, 'true_labels', dataset_names)


def plot_2d_cluster(data, preds, path, name_of_dataset):
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
    plt.scatter(data[i][:, 0], data[i][:, 1], s=10, color=colors[preds[i]-1]) # -1 to colour outliers as black
    plt.title(name_of_dataset[i])
    plt.xticks(())
    plt.yticks(())
  plt.suptitle('True labels', size=20)
  plt.show()
  plt.savefig(path)

plot_true_labels()