
#!/usr/bin/python3
#-*-coding: utf-8-*-

import numpy as np
import pandas as pd
import os

FILENAME = "COAP.pcapng"
PACKET_LENGTH = 300*8 #1365*8 #10920

DATA_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/csv/"
DATA_PATH = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"
SAVE_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"


# Aggregate file
def aggregate_df_file(filename, path, file_type='.txt'):
  """Agregate DataFrame.

  Args:
      filename (str): filename.
      path (str): folder path containing the .csv files.
      file_type (str, optional): files extension. 
      Defaults to '.txt'.

  Returns:
      pd.DataFrame: DataFrame, the aggregation of all 
      the .csv files in the path folder.
  """
  df = pd.DataFrame()
  for f in os.listdir(path):
    print(f)
    if ((filename in f) and (file_type in f)):
      df_tmp = pd.read_csv(path + f)
      df = pd.concat([df, df_tmp], axis=0)
  return df

df_result = aggregate_df_file(filename=FILENAME, 
                              path=DATA_DIR, 
                              file_type='.csv')
num = df_result['num_packet'].values.max() + 1

df_result = df_result.sort_values(by='num_packet', 
                                  ascending=True)
df_result = df_result.reset_index(drop=True)
df_result.to_csv(f"{SAVE_DIR}df_{num}_{FILENAME}.csv", 
                 index=False)
