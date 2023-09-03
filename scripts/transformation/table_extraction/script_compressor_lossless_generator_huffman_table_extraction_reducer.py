#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

# Filter warnings
#import warnings
#warnings.filterwarnings("ignore")

# Garbage collector
import gc

# Folder manipulation
import os

# Linear algebra and data processing
import numpy as np
import pandas as pd
import math
import random

# Get version
from platform import python_version

# Personnal functions
# import functions


#############################################
# SET PARAMETERS
#############################################


HUFFMAN_DIR = "MODELS/HUFFMAN/DEEP_LEARNING/"
HUFFMAN_PARTS_DIR = HUFFMAN_DIR + "PARTS/"
RESULTS_DIR = "RESULTS/"
MODELS_DIR = "MODELS/DEEP_LEARNING/"
MAIN_DIR = "./DATA/"

PROTO = "HTTP"

# Context
LOOK_BACK_CONTEXT = 1 # On rajoute +1 car on prend le dernier paquet comme paquet à compresser...
LOOK_AHEAD_CONTEXT = 1 #TIMESTEPS
CONTEXT_SIZE = LOOK_BACK_CONTEXT # Nombre timesteps sur les contexts
CONTEXT_OUTPUT_SIZE = 30 # Size of contexte in model layer
# De preference < 128 car c'est la taille de la
# couche GRU avant

# Packet
LOOK_BACK_PACKET = 16
LOOK_AHEAD_PACKET = 1
QUANTITY_PACKET = 15000
NUM_SIN = 8
NUM_FEATS = NUM_SIN + 1

# Filter equipment
NB_EQUIP_MAX = 10 # Set to None if no equipment
NB_EQUIP_TRAIN = 5 # Set to None if no equipment
MODE_EQUIP = "train" #"train" # or "test" or None for both
# We create Huffman on train data !
NB_EQUIP_LIMIT = NB_EQUIP_MAX # Limit to check to 
# see if we select the right number
# of equipment

# Generator
SHUFFLE = False #True
ALPHABET_SIZE = 2
BATCH_SIZE = 1 #512 #2048

# Size added to header_length IN BYTES
EXTRA_SIZE = 0
CUSTOM_SIZE = 100 # Take the lead if extra size is define
CHECKSUM = True # True = Keep Checksum

# For huffman creation
CUT_VALUE = 16 # For array of index possition
KEEP_ERROR = True # Change EXT_NAME, remove KEEP ERROR !
OPTIMAL = True
SELECTIVE = True

# Generated dataset parameters
LEFT_PADDING = True # Padding dataset


# Name
# min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE) if CONTEXT_OUTPUT_SIZE == 0
# else LOOK_BACK_CONTEXT == 1 or 2 or 3 or 4...etc
# easy to set LOOK_BACK_CONTEXT == 0
FULL_NAME = f"LOSSLESS_CONTEXT{min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE)}_PACKET{LOOK_BACK_PACKET}_SIN{NUM_SIN}_{PROTO}"


if (CHECKSUM):

    if (CUSTOM_SIZE is not None):
        EXT_NAME = f"_WITH_CHECKSUM_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_EQUIP{MODE_EQUIP}"
    else:
        EXT_NAME = f"_WITH_CHECKSUM_EXTRA_SIZE{EXTRA_SIZE}_MODE_EQUIP{MODE_EQUIP}"

else:

    if (CUSTOM_SIZE is not None):
        EXT_NAME = f"_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_EQUIP{MODE_EQUIP}"
    else:
        EXT_NAME = f"_EXTRA_SIZE{EXTRA_SIZE}_MODE_EQUIP{MODE_EQUIP}"


print(f"MODEL : {FULL_NAME}{EXT_NAME}")

# Set the Seed for numpy random choice
np.random.seed(42)



#############################################
# LAUNCH REDUCER
#############################################



files = os.listdir(f"{HUFFMAN_PARTS_DIR}")
df_all = pd.DataFrame()
num_file_all = []

# Define maxs
max_context_size = 0
max_rank_pos = 0

files_exist = False 

for f in files:
    
    # Extract name info in filename
    metaname = f.split('_')[2:-2]
    metaname = "_".join(metaname)

    if (KEEP_ERROR):
        ref_name = f"{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}"
    else:
        ref_name = f"{FULL_NAME}{EXT_NAME}_{CUT_VALUE}"

    if (SELECTIVE):
        ref_name = ref_name + "_SELECTIVE"

    if (OPTIMAL):
        ref_name = ref_name + "_OPTIMAL"

    if (LEFT_PADDING):
        ref_name = ref_name + "_LEFT_PADDING" 
        
    print("[DEBUG] metaname: ", metaname)
    print("[DEBUG] ref_name: ", ref_name)

    # If filename is part of {FULL_NAME}{EXT_NAME}
    if (metaname == ref_name):

        # Extract num files
        context_size = int(f.split('_')[-2])
        rank_pos = int(f.split('_')[-1].split('.')[0])
        
        # Update maxs
        max_context_size = max(context_size, max_context_size)
        max_rank_pos = max(rank_pos, max_rank_pos)
        
        print("[DEBUG] rank_pos // context_size: ", rank_pos, 
            " // ", context_size)
    
        # Load dataframe
        df = pd.read_csv(f"{HUFFMAN_PARTS_DIR}{f}")
        print("[DEBUG] df.shape: ", df.shape)
        #df = df.rename(columns={"Unnamed: 0": "index_batch"})\
        #            .set_index("index_batch")
        #df['num_file'] = num_file
        
        # Concat with other dataframe
        df_all = pd.concat(
            [df_all, df], axis=0)
        
        # Remove files
        os.remove(f"{HUFFMAN_PARTS_DIR}{f}")

        # File(s) exist
        files_exist = True


#############################################
# SAVE DATAFRAME
#############################################

if (files_exist):

    
    # Keep error or not
    if (KEEP_ERROR):
        df_name = f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}"
    else:
        df_name = f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}"

    # Selective or not
    if (SELECTIVE):
        df_name = df_name + "_SELECTIVE"

    # Selective or not
    if (OPTIMAL):
        df_name = df_name + "_OPTIMAL"

    if (LEFT_PADDING):
        df_name = df_name + "_LEFT_PADDING" 

    # Set num
    df_name = df_name + f"_{max_context_size}_{max_rank_pos}.csv"

    # Save dataframe
    df_all.to_csv(df_name, index=False)


#if (KEEP_ERROR):
    #df_huffman.to_csv(f"{HUFFMAN_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}.csv", index=False)
    #np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}.npy")
#else:
    #df_huffman.to_csv(f"{HUFFMAN_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}.csv", index=False)
    #np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}.npy")


