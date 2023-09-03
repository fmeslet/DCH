#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os

FILENAME = "COAP.pcapng"
PACKET_LENGTH = 300*8 # 1365*8 = 10920

DATA_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/npy/"
DATA_PATH = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"
SAVE_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"


def bubble_sort_filename(path, filename):
  """Sort array by filename.

  Args:
      path (str): path of array to sort.
      filename (str): name contained if 
      the file to load.

  Returns:
      list: filenames sorted.
  """
  result = []
  files = os.listdir(path)
  # files.remove('sample_data')
  min = 9e9
  for i in range(len(files)+1):
    for f in files:
      # Search for the digit minimm AND 
      # with the corresponding file name
      if ((len(f.split('_')) > 1) and 
          (filename in f) and 
          ('arr' in f)):
        num = int(f.split('_')[1])
        if (num < min):
          min = num
          f_min = f

    if(f_min != ""):
      result.append(f_min)
      files.remove(f_min)
    else:
      return result

    # Re init constant
    min = 9e9
    f_min = ""

# Aggregate file
def aggregate_arr_file(filename, path, 
                       files, file_type='.txt'):
  """Agregate Numpy array.

  Args:
      filename (str): filename.
      path (str): folder path containing the .csv files.
      files (list): list of filename ordered.
      file_type (str, optional): files extension. 
      Defaults to '.txt'.

  Returns:
      pd.DataFrame: DataFrame, the aggregation of all 
      the .csv files in the path folder.
  """
  arr = np.empty(
     (0, PACKET_LENGTH)).astype(np.uint8)
  for f in files:
    print(f)
    if ((filename in f) and (file_type in f)):
      arr_tmp = np.load(path + f).astype(np.uint8)
      arr = np.concatenate((arr, arr_tmp), axis=0)
      arr = arr.astype(np.uint8)
      #df = df.reset_index(drop=True)
  return arr

result = bubble_sort_filename(path=DATA_DIR, filename=FILENAME)
result_arr = aggregate_arr_file(filename=FILENAME, files=result, path=DATA_DIR, file_type='.npy')
num = result_arr.shape[0]

np.save(file=f"{SAVE_DIR}arr_{num}_{FILENAME}.npy", arr=result_arr)

# Supprimer les arrays
