#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

# Garbage collector
import gc
import os

# Linear algebra and data processing
import numpy as np
import pandas as pd
import math
import random

# Get version
from platform import python_version


#############################################
# SET PARAMETERS
#############################################


FIELDS_DIR = "RESULTS/FIELDS/DEEP_LEARNING/PARTS/"
MODELS_DIR = "MODELS/DEEP_LEARNING/"
MAIN_DIR = "./DATA/"

PROTO = "COAP"

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
QUANTITY_PACKET = 15000 # Check if similar to training !
NUM_SIN = 8
NUM_FEATS = NUM_SIN + 1

# For Occlusion Map
NUM_RANK_FIELDS = 29 # Num of rank used for occlusion
KEEP_POS = False # Keep Sinusoide as a field ?
QUANTITY_BATCH = 10 # Num of batch used in total 
# If None all batch in train set is used

# Filter equipment
MODE_EQUIP = "train" # or "test" or None for both

# Size added to header_length IN BYTES
EXTRA_SIZE = 0
CUSTOM_SIZE = 100
CHECKSUM = True # True = Keep Checksum

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


# If padding on dataset
if (LEFT_PADDING):
    EXT_NAME += f"_LEFT_PADDING"


print(f"MODEL : {FULL_NAME}{EXT_NAME}")

# Set the Seed for numpy random choice
np.random.seed(42)


#############################################
# LAUNCH TRAINING
#############################################


files = os.listdir(f"{FIELDS_DIR}")
df_all = pd.DataFrame()
num_file_all = []
max_num_file = 0

files_exist = False 

for f in files:

    # Extract name info in filename
    metaname = f.split('_')[2:-1]
    metaname = "_".join(metaname)
    #print("[DEBUG] filename: ", metaname, 
    #        " // ", num_file)

    # If filename is part of {FULL_NAME}{EXT_NAME}
    if (metaname == f"{FULL_NAME}{EXT_NAME}"):
    
        # Extract num files
        num_file = int(f.split('_')[-1].split('.')[0])
        max_num_file = max(num_file, max_num_file)
        print("[DEBUG] filename: ", metaname,
              " // ", num_file)

        # Load dataframe
        df = pd.read_csv(f"{FIELDS_DIR}{f}")
        df = df.rename(columns={"Unnamed: 0": "index_batch"})\
                    .set_index("index_batch")
        df['num_file'] = num_file
        
        # Concat with other dataframe
        df_all = pd.concat(
            [df_all, df], axis=0)
        
        # Remove files
        os.remove(f"{FIELDS_DIR}{f}")

        # File(s) exist
        files_exist = True


#############################################
# SAVE DATAFRAME
#############################################


if (files_exist):
    df_all.to_csv(f"{FIELDS_DIR}/df_FIELDS_{FULL_NAME}{EXT_NAME}_{max_num_file}.csv", index=True)

#if (files_exist):
#    df_all.to_csv(f"{RESULTS_DIR}/df_FIELDS_{FULL_NAME}{EXT_NAME}_{max_num_file}.csv", index=True)
