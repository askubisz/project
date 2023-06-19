from threadpoolctl import threadpool_limits

# Import module from parent directory (https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from synthethic_benchmark import benchmark_function


with threadpool_limits(limits=1, user_api='blas'):
  dbscan_time, dbscan_ar, dbscan_ami = benchmark_function(cluster_function='dbscan')[:3]

dbscan_time.to_csv("dbscan/dbscan_time.csv")
dbscan_ar.to_csv("dbscan/dbscan_ar.csv")
dbscan_ami.to_csv("dbscan/dbscan_ami.csv")