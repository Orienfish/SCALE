import numpy as np
import pickle
import os

# checks if directory exists; makes new directory if not exist
# returns directory name
def smart_dir(dir_name, base_list = None):
    dir_name = dir_name + '/'
    if base_list is None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name
    else:
        dir_names = []
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for d in range(len(base_list)):
            dir_names.append(dir_name + base_list[d] + '/')
            if not os.path.exists(dir_names[d]):
                os.makedirs(dir_names[d])
        return dir_names
