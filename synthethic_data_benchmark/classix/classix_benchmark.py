from threadpoolctl import threadpool_limits

# Import module from parent directory (https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from synthethic_benchmark import benchmark_function

# Requires around 12GB RAM
with threadpool_limits(limits=1, user_api='blas'):
  classix_time, classix_ar, classix_ami = benchmark_function(cluster_function='classix')[:3]

classix_time.to_csv("classix/classix_time.csv")
classix_ar.to_csv("classix/classix_ar.csv")
classix_ami.to_csv("classix/classix_ami.csv")