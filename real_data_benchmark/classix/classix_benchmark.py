from threadpoolctl import threadpool_limits

# Import module from parent directory (https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369)
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from real_data_benchmark import real_data_benchmark


with threadpool_limits(limits=1, user_api='blas'):
  df = real_data_benchmark('classix')

df.to_csv("classix/results.csv")