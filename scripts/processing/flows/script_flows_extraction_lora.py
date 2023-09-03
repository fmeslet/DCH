#!/usr/bin/python3
#-*-coding: utf-8-*-

import scapy_layers

import gc
import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP, UDP, IP, Padding, Raw, load_layer, Ether, CookedLinux, PcapReader
from scapy.compat import bytes_encode

load_layer("http")

MAIN_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"
PROCESS_DIR = MAIN_DIR + "PROCESS/"

PROTO = "LORA"
NB_FILES = 20

FILENAME = f"LORA_{NB_FILES}"


#############
# FUNCTIONS
#############


################
# PROCESS DATA
################

# Load data

data_raw = pd.read_csv(f"{MAIN_DIR}df_raw_{PROTO}_{NB_FILES}.csv")
arr_raw = np.load(f"{MAIN_DIR}arr_{PROTO}_{NB_FILES}_bit.npy", mmap_mode='r')


# Change columns names
data_raw['length_total'] = data_raw['size_array']
data_raw['min_header_length'] = int((3+3+2+32+8+16+0+8)/8)
data_raw['header_length'] = \
        data_raw['min_header_length'] + \
        data_raw['fopts_size'].fillna(0)


# Fill NaN and set zeros and remove 
# CRC status = -1

# Remove packet with wrong CRC
condition = (data_raw['crc_status'] == 1)
indexes = data_raw[condition].index.values
data_raw = data_raw[condition]\
            .reset_index(drop=True)

max_length = int(
    data_raw["length_total"].max() * 8)
arr = arr_raw[
    indexes, :max_length]

data_test = data_raw.copy()


# Extract 

columns = ['device_address', 'gateway']

print("COLUMNS : ", data_raw.columns)

sessions = data_test.groupby( #[condition]
    columns).size().reset_index().rename(
    columns={0:'count'})
sessions = sessions.copy()

for i in range(sessions.shape[0]): #result.shape[0]
    condition_flow = (((data_test['device_address'] == sessions['device_address'].iloc[i]) &
                       (data_test['gateway'] == sessions['gateway'].iloc[i])))
    # Add flow id
    data_test.loc[condition_flow, "flow_id_mac"] = i

    if(data_test[condition_flow].shape[0] <= 1):
        print(sessions.iloc[i])
    

# flows = data_test[
#    'flow_id'].value_counts().sort_index().index.values
# data_test.to_csv(f"{RESULTS_DIR}DF_{TYPE}_FLOWS_{PROTO}_FINAL_TMP.csv", index=False)


columns = ['device_address', 'gateway', 'fport']

data_test.at[data_test['fport'].isna(), "fport"] = -2

sessions = data_test.groupby( #[condition]
    columns).size().reset_index().rename(
    columns={0:'count'})
sessions = sessions.copy()

for i in range(sessions.shape[0]): #result.shape[0]
    condition_flow = (((data_test['device_address'] == sessions['device_address'].iloc[i]) &
                       (data_test['fport'] == sessions['fport'].iloc[i]) &
                       (data_test['gateway'] == sessions['gateway'].iloc[i])))
    # Add flow id
    data_test.loc[condition_flow, "flow_id"] = i

    if(data_test[condition_flow].shape[0] <= 1):
        print(sessions.iloc[i])
    

# Set timesteps and header length

# Convert to float
data_test['timestamps'] = data_test['time']\
        .apply(lambda x: pd.to_datetime(x).timestamp())


data_test.to_csv(f"{PROCESS_DIR}df_process_{PROTO}_{NB_FILES}.csv", index=False)
np.save(arr=arr, file=f"{PROCESS_DIR}arr_process_{PROTO}_{NB_FILES}_bit.npy")

