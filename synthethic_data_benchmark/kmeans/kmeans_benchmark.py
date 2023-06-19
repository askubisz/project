from threadpoolctl import threadpool_limits

# Import module from parent directory (https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from synthethic_benchmark import benchmark_function


with threadpool_limits(limits=1, user_api='blas'):
  kmeans_time, kmeans_ar, kmeans_ami, kmeans_iter = benchmark_function()

kmeans_time.to_csv("kmeans/kmeans_time.csv")
kmeans_ar.to_csv("kmeans/kmeans_ar.csv")
kmeans_ami.to_csv("kmeans/kmeans_ami.csv")
kmeans_iter.to_csv("kmeans/kmeans_iter.csv")