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

PROTO = "SNMP"


#############
# FUNCTIONS
#############


def process_data(df):
    """Process DataFrame to remove null values and
    compute various length.

    Args:
        df (pd.DataFrame): data about colected packets.

    Returns:
        pd.DataFrame: processed DataFrame.
    """
    
    # Fill NaN in Layers columns
    df.loc[df['layers_2'].isna(), 'layers_2'] = "None"
    df.loc[df['layers_3'].isna(), 'layers_3'] = "None" 
    df.loc[df['layers_4'].isna(), 'layers_4'] = "None" 
    df['layers_5'] = "None"
    #df.loc[df['layers_5'].isna(), 'layers_5'] = "None"

    df.loc[df['length_2'].isna(), 'length_2'] = 0
    df.loc[df['length_3'].isna(), 'length_3'] = 0
    df.loc[df['length_4'].isna(), 'length_4'] = 0
    df['length_5'] = 0
    #df.loc[df['length_5'].isna(), 'length_5'] = 0

    df.loc[df['ip_src'].isna(), 'ip_src'] = "None"
    df.loc[df['ip_dst'].isna(), 'ip_dst'] = "None" 
    df.loc[df['flags'].isna(), 'flags'] = 0 
    df.loc[df['sport'].isna(), 'sport'] = 0
    df.loc[df['dport'].isna(), 'dport'] = 0
    
    # GET LENGTHS
    for i in reversed(range(0, 6)):
        #print(f"Layers {i}")
        condition = (df[f"layers_{i}"] == "None")
        df.loc[condition, "payload_layers"] = i
        if(i > 0):
            df.loc[condition, "header_length"] = df[
                    condition]["length_total"] - df[condition][f"length_{i-1}"]
    df["payload_length"] = df["length_total"] - df["header_length"]
    
    # Drop les 128 paquets de Mardi qui servent à rien...
    condition_drop = (df['application'] == 'nan')
    df = df.drop(df[condition_drop].index.values)
    df = df.reset_index(drop=True)

    return df


################
# PROCESS DATA
################

# Load data

data_raw = pd.read_csv(f"{MAIN_DIR}df_raw_{PROTO}.csv")
arr_raw = np.load(f"{MAIN_DIR}arr_{PROTO}_bit.npy", mmap_mode='r')


# Fill NaN and set zeros

data_test = process_data(
        data_raw.copy())


# Update layers and length


diff_layer_size = (data_test['length_0'] - \
        data_test['length_1']).astype(int)

# Remove layers_0 and length_0
data_test = data_test.drop(
    ['layers_0', 'length_0'], axis=1)   

# Update name of layers
for i in range(1, 6):
    
    # Rename columns 
    data_test = data_test.rename(
        columns={f"layers_{i}":f"layers_{i-1}"})
    data_test = data_test.rename(
        columns={f"length_{i}":f"length_{i-1}"})
    
# Remove length to header_length and length_total
data_test['length_total'] = data_test['length_total'] - diff_layer_size  
data_test['header_length'] = data_test['header_length'] - diff_layer_size  
data_test['payload_layers'] = data_test['payload_layers'] - 1


# Update arr_raw
start_idx = int(diff_layer_size.max()*8)
arr = arr_raw[:, start_idx:]


# Compute max length and update array
max_length = int(
    (data_test['length_total'].max()*8))
arr = arr[:, :max_length]


# Extract 

columns = ['ip_src', 'ip_dst', 'sport', 
           'dport', 'application']

print("COLUMNS : ", data_raw.columns)

sessions = data_test.groupby(
    columns).size().reset_index().rename(
    columns={0:'count'})
sessions = sessions[(sessions['sport'] != 0) & 
                    (sessions['dport'] != 0)].copy()

for i in range(sessions.shape[0]):
    condition_flow = (((data_test['ip_src'] == sessions['ip_src'].iloc[i]) &
                     (data_test['ip_dst'] == sessions['ip_dst'].iloc[i]) & 
                     (data_test['sport'] == sessions['sport'].iloc[i]) &
                     (data_test['dport'] == sessions['dport'].iloc[i]) &
                     (data_test['application'] == sessions['application'].iloc[i])) | 
                    ((data_test['ip_src'] == sessions['ip_dst'].iloc[i]) &
                     (data_test['ip_dst'] == sessions['ip_src'].iloc[i]) & 
                     (data_test['sport'] == sessions['dport'].iloc[i]) &
                     (data_test['dport'] == sessions['sport'].iloc[i]) &
                     (data_test['application'] == sessions['application'].iloc[i])))
    # Add flow id
    data_test.loc[condition_flow, "flow_id"] = i

    if(data_test[condition_flow].shape[0] <= 1):
        print(sessions.iloc[i])


# flows = data_test[
#    'flow_id'].value_counts().sort_index().index.values
# data_test.to_csv(f"{RESULTS_DIR}DF_{TYPE}_FLOWS_{PROTO}_FINAL_TMP.csv", index=False)


# In case of no IP layer

start_index = int(data_test['flow_id'].max() + 1)
columns = ['mac_src', 'mac_dst', 'layers_1', 'application']

sessions = data_test.groupby(
    columns).size().reset_index().rename(
    columns={0:'count'})

for i in range(sessions.shape[0]):
    condition_flow = (((data_raw['mac_src'] == sessions['mac_src'].iloc[i]) &
                      (data_raw['mac_dst'] == sessions['mac_dst'].iloc[i]) &
                      (data_raw['layers_1'] == sessions['layers_1'].iloc[i]) & 
                      (data_raw['application'] == sessions['application'].iloc[i]) &
                      (data_raw['sport'] == 0) & (data_raw['dport'] == 0)) |
                    ((data_raw['mac_src'] == sessions['mac_dst'].iloc[i]) &
                     (data_raw['mac_dst'] == sessions['mac_src'].iloc[i]) &
                     (data_raw['layers_1'] == sessions['layers_1'].iloc[i]) & 
                     (data_raw['application'] == sessions['application'].iloc[i]) &
                     (data_raw['sport'] == 0) & (data_raw['dport'] == 0)))
        
    # Add flow id
    data_test.loc[
        condition_flow, "flow_id"] = i+start_index

    #start_index = start_index+sessions.shape[0]


# IP only

columns = ['ip_src', 'ip_dst', 'application']

sessions = data_test.groupby(
    columns).size().reset_index().rename(
    columns={0:'count'})
sessions = sessions.copy()

for i in range(sessions.shape[0]):
    condition_flow = (((data_test['ip_src'] == sessions['ip_src'].iloc[i]) &
                       (data_test['ip_dst'] == sessions['ip_dst'].iloc[i]) &
                       (data_test['application'] == sessions['application'].iloc[i])) | 
                     ((data_test['ip_src'] == sessions['ip_dst'].iloc[i]) &
                       (data_test['ip_dst'] == sessions['ip_src'].iloc[i]) &
                       (data_test['application'] == sessions['application'].iloc[i])))
    # Add flow id
    data_test.loc[
           condition_flow, "flow_id_ip"] = i

    #if(data_test[condition_flow].shape[0] <= 1):
    #    print(sessions.iloc[i])


# Mac only

columns = ['mac_src', 'mac_dst', 'application']

sessions = data_test.groupby(
    columns).size().reset_index().rename(
    columns={0:'count'})
sessions = sessions.copy()

for i in range(sessions.shape[0]):
    condition_flow = (((data_test['mac_src'] == sessions['mac_src'].iloc[i]) &
                       (data_test['mac_dst'] == sessions['mac_dst'].iloc[i]) &
                       (data_test['application'] == sessions['application'].iloc[i])) | 
                     ((data_test['mac_src'] == sessions['mac_dst'].iloc[i]) &
                       (data_test['mac_dst'] == sessions['mac_src'].iloc[i]) &
                       (data_test['application'] == sessions['application'].iloc[i])))
    # Add flow id
    data_test.loc[
           condition_flow, "flow_id_mac"] = i

    #if(data_test[condition_flow].shape[0] <= 1):
    #    print(sessions.iloc[i])
    
    
data_test.to_csv(f"{PROCESS_DIR}df_process_{PROTO}.csv", index=False)
np.save(arr=arr, file=f"{PROCESS_DIR}arr_process_{PROTO}_bit.npy")

