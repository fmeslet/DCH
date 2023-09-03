#!/usr/bin/python3
#-*-coding: utf-8-*-

import scapy_layers

import gc
import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP, UDP, IP, Padding, Raw, load_layer, Ether, CookedLinux, PcapReader, IPv6
from scapy.contrib.mqtt import MQTT
from scapy.contrib.coap import CoAP
from scapy.compat import bytes_encode
import base64

load_layer("http")
#load_layer("https")

FILENAME = "MQTT_IEEE.pcap"
NB_FILES = 20

DATA_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/DATA_RAW/LORA/"
DATA_PATH = DATA_DIR + FILENAME
SAVE_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"


def array_base64_to_bin(x):
    """Convert data in byte format to binary.

    Args:
        x (np.array): data to convert in byte format.

    Returns:
        np.array: data converted to binary format.
    """
    x_decoded = base64.decodebytes(x)
    x_byte_list = ["{:08b}".format(x) for x in x_decoded]
    x_bit_list = []
    for byte in x_byte_list:
        for bit in byte:
            x_bit_list.append(bit)
    
    x_new = np.array(
        x_bit_list, dtype=np.uint8)
    return x_new


def array_bin_to_int(x, size=8):
    """Convert data in binary to integer.
    Args:
        x (np.array): data to convert in binary format.
        size (int, optional): number of bit to compute 
        the integer. Defaults to 8.

    Returns:
        np.array: data converted to integer format.
    """
    x_new = []    
    for i in range(
            0, x.size, size):
        value = 0
        for k, j in enumerate(reversed(range(size))):
            value += x[i+k]*(2**j)
            
        x_new.append(value)
    x_new = np.array(x_new)
    return x_new


def load_data(nb_files):
    """Load the data.

    Args:
        nb_files (int): number of file to load.

    Returns:
        tuple: DataFrame with information 
        and array with data.
    """
    
    filenames = os.listdir(
        DATA_DIR)
    print("[DEBUG][load_data] filenames[:nb_files]: ", filenames[:nb_files])
    data = pd.DataFrame()
    
    nb_rows = 0
    nb_cols = 0
    size_header_bit = 3+3+2+32
    for f in filenames[0:nb_files]:
        
        # Load array
        data_tmp = pd.read_csv(
            DATA_DIR+f)
        nb_rows += data_tmp.shape[0]
        nb_cols = max(
            nb_cols, 
            data_tmp['size'].max()+\
                size_header_bit)
        data = pd.concat(
            [data, data_tmp], axis=0)
    
    # Create data array
    data = data.reset_index(
        drop=True)
    data["size_array"] = 0
    array = np.zeros(
        (nb_rows, 3000)) # nb_cols*8 for bit format
    idx = 0
    
    for f in filenames[0:nb_files]:

        # Load array
        data_tmp = pd.read_csv(
            DATA_DIR+f)
        
        for i in range(
            data_tmp.shape[0]):
            
            # Check CRC status
            crc_status = data_tmp[
                'crc_status'].iloc[i]
            
            # Extract payload
            physical_payload = data_tmp[
                'physical_payload'].iloc[i]
            
            #print("[DEBUG] type(physical_payload): ", 
            #      type(physical_payload), 
            #      " // ", physical_payload)
            
            if (not isinstance(
                physical_payload, float)):
                
                # Extract physical payload
                physical_payload_byte = bytes(
                    physical_payload.encode('utf-8'))
            
                # Bit conversion
                physical_payload_bit = array_base64_to_bin(
                    physical_payload_byte, size=8)
                array_tmp = physical_payload_bit

                # Fopt size
                fopts = array_tmp[8:-32][:32+8][-4:]
                #print("[DEBUG] array_tmp[8:-32][:32+8] : ", array_tmp[8:-32][:32+8])
                try:
                    fopts_size = array_bin_to_int(
                         x=fopts, size=4)[0]
                except:
                    fopts_size = np.NaN
            
                # Add to array
                array[idx,:array_tmp.shape[-1]] = array_tmp[:,]
                
                # Set size array and fopt size
                data.at[idx, "size_array"] = int(array_tmp.size / 8)
                data.at[idx, "fopts_size"] = fopts_size             
            

            # Update idx
            idx += 1
        
    return data, array



# LOAD DATA
data, array = load_data(
    nb_files=NB_FILES) # AVEC NB_FILES...

# SAVE DATA and ARRAY
data.to_csv(f"{SAVE_DIR}df_raw_LORA_{NB_FILES}.csv", index=False)
np.save(f"{SAVE_DIR}arr_LORA_{NB_FILES}_bit.npy", array)
