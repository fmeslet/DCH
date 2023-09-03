#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

# Garbage collector
import gc

# Others
import sys

# Linear algebra and data processing
import numpy as np
import pandas as pd
import math
import random

# Get version
from platform import python_version
import sklearn
import tensorflow as tf

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

# Time measure
import datetime

# Multiprocessing
from multiprocessing import Pool
import multiprocessing as mp


#############################################
# SET PARAMETERS
#############################################


HUFFMAN_DIR = "MODELS/HUFFMAN/INTERPRETATION/"
RESULTS_DIR = "RESULTS/HUFFMAN/INTERPRETATION/"
MODELS_DIR = "MODELS/DEEP_LEARNING/"
MAIN_DIR = "./DATA/"

# Set keras types
tf.keras.backend.set_floatx('float64')

# Context
LOOK_BACK_CONTEXT = 1 # On rajoute +1 car on prend le dernier paquet comme paquet à compresser...
LOOK_AHEAD_CONTEXT = 1 #TIMESTEPS
CONTEXT_SIZE = 2 # Nombre timesteps sur les contexts
CONTEXT_OUTPUT_SIZE = 30 # Size of contexte in model layer
# De preference < 128 car c'est la taille de la
# couche GRU avant

# Packet
LOOK_BACK_PACKET = 8
LOOK_AHEAD_PACKET = 1
NUM_SIN = 8
NUM_FEATS = NUM_SIN + 1

# Learning parameters
SHUFFLE = False # For test generator
ALPHABET_SIZE = 2
BATCH_SIZE = 1 # For test generator

# Size added to header_length IN BYTES
EXTRA_SIZE = 0
CUSTOM_SIZE = 100 # Take the lead if extra size is define
CHECKSUM = True # True = Keep Checksum

# For evaluation parallelisation
BATCH_PACKET = 20

# For Huffman model only
CUT_VALUE = 4
KEEP_ERROR = True
OPTIMAL = True
SELECTIVE = True # if True he most rank bit is used (only one)
SOFT_MODE = False # True par default permet de prendre la moyenne si le champs est meconnu !
DECIMALS = 9 # Si None pas de rounding !

# Generated dataset parameters
MAX_PACKET_RANK = 3 # Max rank support for packet
NB_BITS = 16*2 # Multiple for field inversion
NB_ROWS = 4000*MAX_PACKET_RANK # Multiple for the flow id NB_ROWS = x * MAX_PACKET_RANK
MODE_DATASET = "combinaison" # "random", "checksum", "checksum34", "fixed", "inversion", "counter", "fixed_flow", "combinaison"
LEFT_PADDING = True # Padding dataset

# Name
# min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE) if CONTEXT_OUTPUT_SIZE == 0
# else LOOK_BACK_CONTEXT == 1 or 2 or 3 or 4...etc
# easy to set LOOK_BACK_CONTEXT == 0
FULL_NAME = f"LOSSLESS_CONTEXT{min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE)}_PACKET{LOOK_BACK_PACKET}_SIN{NUM_SIN}"

if (CHECKSUM):
        
    if (KEEP_ERROR):

        if (CUSTOM_SIZE is not None):
            EXT_NAME = f"_HUFFMAN_WITH_CHECKSUM_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}_KEEP_ERROR_{CUT_VALUE}_DECIMALS{DECIMALS}"
        else:
            EXT_NAME = f"_HUFFMAN_WITH_CHECKSUM_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}_KEEP_ERROR_{CUT_VALUE}_DECIMALS{DECIMALS}"

    else:

        if (CUSTOM_SIZE is not None):
            EXT_NAME = f"_HUFFMAN_WITH_CHECKSUM_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}_{CUT_VALUE}_DECIMALS{DECIMALS}"
        else:
            EXT_NAME = f"_HUFFMAN_WITH_CHECKSUM_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}_{CUT_VALUE}_DECIMALS{DECIMALS}"

else:
    
    if (KEEP_ERROR):

        if (CUSTOM_SIZE is not None):
            EXT_NAME = f"_HUFFMAN_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}_KEEP_ERROR_{CUT_VALUE}_DECIMALS{DECIMALS}"
        else:
            EXT_NAME = f"_HUFFMAN_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}_KEEP_ERROR_{CUT_VALUE}_DECIMALS{DECIMALS}"

    else:

        if (CUSTOM_SIZE is not None):
            EXT_NAME = f"_HUFFMAN_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}_{CUT_VALUE}_DECIMALS{DECIMALS}"
        else:
            EXT_NAME = f"_HUFFMAN_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}_{CUT_VALUE}_DECIMALS{DECIMALS}"
                

# If the SELECTIVE mode is activated 
if (SELECTIVE):
    EXT_NAME = EXT_NAME + "_SELECTIVE"


# If the OPTIMAL mode is activated 
if (OPTIMAL):
    EXT_NAME = EXT_NAME + "_OPTIMAL"


# If the LEFT_PADDING mode is activated 
if (LEFT_PADDING):
    EXT_NAME = EXT_NAME + "_LEFT_PADDING"


# Set the Seed for numpy random choice
np.random.seed(42)


#############################################
# USEFULL CLASS/FUNCTIONS
#############################################

class ArithmeticCoderBase(object):
    
    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, statesize):
        
        self.STATE_SIZE = statesize
        self.MAX_RANGE = 1 << self.STATE_SIZE
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        self.MAX_TOTAL = self.MIN_RANGE
        self.MASK = self.MAX_RANGE - 1
        self.TOP_MASK = self.MAX_RANGE >> 1
        self.SECOND_MASK = self.TOP_MASK >> 1
        
        self.low = 0
        self.high = self.MASK

    def update(self,  cumul, symbol):
        # State check
        low = self.low
        high = self.high
        range_value = high - low + 1
            
        # Frequency table values check
        total = cumul[-1].item() #np.asscalar(cumul[-1])
        symlow = cumul[symbol].item()  #np.asscalar(cumul[symbol])
        symhigh = cumul[symbol+1].item() #np.asscalar(cumul[symbol+1])
        
        # Update range
        newlow  = low + symlow  * range_value // total
        newhigh = low + symhigh * range_value // total - 1
        self.low = newlow
        self.high = newhigh
        # While the highest bits are equal
        while ((self.low ^ self.high) & self.TOP_MASK) == 0:
            self.shift()
            self.low = (self.low << 1) & self.MASK
            self.high = ((self.high << 1) & self.MASK) | 1
        
        # While the second highest bit of low is 1 and the second highest bit of high is 0
        while (self.low & ~self.high & self.SECOND_MASK) != 0:
            self.underflow()
            self.low = (self.low << 1) & (self.MASK >> 1)
            self.high = ((self.high << 1) & (self.MASK >> 1)) | self.TOP_MASK | 1
    
    def shift(self):
        raise NotImplementedError()
    
    def underflow(self):
        raise NotImplementedError()
        
class ArithmeticEncoder(ArithmeticCoderBase):
    
    def __init__(self, statesize, bitout,
                 write_mode=False):
        super(ArithmeticEncoder, self).__init__(statesize)
        self.output = bitout
        self.write_mode = write_mode
        self.num_underflow = 0
        self.data_compress = []
    
    def write(self, cumul, symbol):
        self.update(cumul, symbol)
    
    def finish(self):
        if (self.write_mode):
            self.output.write(1)
        #else:
        #    self.data_compress.append(1)
            
    
    def shift(self):
        bit = self.low >> (self.STATE_SIZE - 1)
        if (self.write_mode):
            self.output.write(bit)
        else:
            self.data_compress.append(bit)
        
        # Write out the saved underflow bits
        for _ in range(self.num_underflow):
            if (self.write_mode):
                self.output.write(bit ^ 1)
            else:
                self.data_compress.append(bit ^ 1)
        self.num_underflow = 0
    
    def underflow(self):
        self.num_underflow += 1


class ArithmeticDecoder(ArithmeticCoderBase):
    
    def __init__(self, statesize, bitin, 
                 data_compress, 
                 write_mode=False):
        super(ArithmeticDecoder, self).__init__(statesize)
        # The underlying bit input stream.
        self.input = bitin
        self.data_compress = data_compress
        self.data_compress.append(1) # Set the EOF here
        # because we measure compression size at the end
        self.data_decompress = []
        self.rank = 0
        self.write_mode = write_mode
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        self.temp_read = []
        self.temp = []
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()
    
    def read(self, cumul, alphabet_size):
        
        total = cumul[-1].item() #np.asscalar(cumul[-1])
        range_value = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range_value
        
        start = 0
        end = alphabet_size
        while end - start > 1:
            middle = (start + end) >> 1
            if cumul[middle] > value:
                end = middle
            else:
                start = middle
        
        symbol = start

        self.update(cumul, symbol)
        self.data_decompress.append(symbol)
        return symbol
    
    
    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()
        
    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()
    
    def read_code_bit(self):
        if (self.write_mode):
            temp = self.input.read()
        else:
            try:
                temp = self.data_compress[self.rank]
            except Exception as e:
                temp = -1
            self.rank += 1
        self.temp.append(temp)
        if temp == -1:
            temp = 0
        return temp

class BitInputStream(object):
    
    # Constructs a bit input stream based on the given byte input stream.
    def __init__(self, inp):
        self.input = inp
        self.currentbyte = 0
        self.numbitsremaining = 0

    def read(self):
        if self.currentbyte == -1:
            return -1
        if self.numbitsremaining == 0:
            temp = self.input.read(1)
            if len(temp) == 0:
                self.currentbyte = -1
                return -1
            self.currentbyte = temp[0] if python3 else ord(temp)
            self.numbitsremaining = 8
        assert self.numbitsremaining > 0
        self.numbitsremaining -= 1
        return (self.currentbyte >> self.numbitsremaining) & 1
    
    def read_no_eof(self):
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()
    
    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


class BitOutputStream(object):
    
    # Constructs a bit output stream based on the given byte output stream.
    def __init__(self, out):
        self.output = out  # The underlying byte stream to write to
        self.currentbyte = 0  # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        self.numbitsfilled = 0  # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
    
    def write(self, b):
        if b not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
        self.currentbyte = (self.currentbyte << 1) | b
        self.numbitsfilled += 1
        if self.numbitsfilled == 8:
            towrite = bytes((self.currentbyte,)) if python3 else chr(self.currentbyte)
            self.output.write(towrite)
            self.currentbyte = 0
            self.numbitsfilled = 0

    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()


def standardize(x, min_x, max_x, a, b):
  # x_new in [a, b]
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new


def train_val_test_split(X, y, random_state, train_size=0, 
                         val_size=0, test_size=0):
    
    # Prendre le cas de la stratification
    # Prendre en cmpte la spération...
    X = np.arange(0, X.shape[0])
    y = np.arange(0, y.shape[0])
    train_idx, val_idx, _, _ = sklearn.model_selection.train_test_split(X, y,
                                random_state=random_state, test_size=1-train_size, 
                                shuffle=True) # , stratify=y

    # Get data test from val
    X = X[val_idx]
    y = y[val_idx]
    val_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(X, y,
                                random_state=random_state, test_size=0.5, 
                                shuffle=True)
    
    return train_idx, val_idx, test_idx


def create_windows(data, window_shape, step = 1, start_id = None, end_id = None):
    
    data = np.asarray(data)
    data = data.reshape(-1,1) if np.prod(data.shape) == max(data.shape) else data
        
    start_id = 0 if start_id is None else start_id
    end_id = data.shape[0] if end_id is None else end_id
    
    data = data[int(start_id):int(end_id),:]
    window_shape = (int(window_shape), data.shape[-1])
    step = (int(step),) * data.ndim
    slices = tuple(slice(None, None, st) for st in step)
    indexing_strides = data[slices].strides
    win_indices_shape = ((np.array(data.shape) - window_shape) // step) + 1
    
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(data.strides))
    
    window_data = np.lib.stride_tricks.as_strided(data, shape=new_shape, strides=strides)
    
    return np.squeeze(window_data, 1)


def gen_sin(F, Fs, phi, A, mean, period, center):
    T = (1/F)*period # Period if 10/F = 10 cycles
    Ts = 1./Fs # Period sampling
    N = int(T/Ts) # Number of samples
    t = np.linspace(0, T, N)
    signal = A*np.sin(2*np.pi*F*t + phi) + center
    return signal


class TableModel():
    def __init__(self, 
                 df_table, 
                 index_pos,
                 alphabet_size):
        self.df_table = df_table
        self.index_pos = index_pos
        self.alphabet_size = alphabet_size
        
    def predict(self, 
                data, 
                pos,
                ctx,
                soft=True):
        
        pred_proba = np.ones(
            (data.shape[0], self.alphabet_size))*0.5
        
        # For each block of a packet
        for i in range(data.shape[0]):

            '''print("")
            print("[DEBUG][predict] i: ", i)
            print("[DEBUG][predict] data.shape: ", data.shape)
            print("[DEBUG][predict] ctx[i]: ", ctx[i])
            print("[DEBUG][predict] pos[i]: ", pos[i])
            print("[DEBUG][predict] self.index_pos[ctx[i], pos[i]]: ", 
                    self.index_pos[ctx[i], pos[i]])
            print("[DEBUG][predict] len(ctx): ", len(ctx))
            print("[DEBUG][predict] len(pos): ", len(pos))
            print("[DEBUG][predict] ctx: ", ctx)
            print("[DEBUG][predict] pos: ", pos)'''
        
            # Extract bits
            # If the position est known else except
            try:
                
                # Extract indexes
                indexes_values_extract = \
                    self.index_pos[
                        ctx[i], pos[i]]
                
                if (OPTIMAL or SELECTIVE):
                    
                    # Remove -1
                    indexes_values_extract = \
                        indexes_values_extract[
                            indexes_values_extract >= 0]
                
                # Extract values
                values_extract = data[
                    i, indexes_values_extract]

                
                if ((OPTIMAL or SELECTIVE) and 
                    (values_extract.size < CUT_VALUE)):
                    
                    # Set 0 at the beginning
                    values_extract = np.lib.pad(values_extract,
                        (CUT_VALUE-values_extract.size, 0),
                        'constant', constant_values=(0))

                # Convert bit to str
                values_extract = values_extract\
                    .ravel().astype(
                        np.uint8).astype(str)

                # Extract bit 
                values_extract = "".join(
                    values_extract)
                
                values_extract = str(values_extract)
                
                # Get values
                index_pos = self.df_table.loc[
                    (ctx[i],)].index.get_level_values(0)
                if (np.isin(pos[i], index_pos).any()):
                    df_tmp = self.df_table.loc[
                        (ctx[i], pos[i], )]
                
                    # Apply cond
                    cond = (values_extract == df_tmp.index.values)
                    is_value = (df_tmp[cond].shape[0] == 0)
                    
                    if (not is_value):
                        val = df_tmp[cond].values
                        #pred_proba[i:i+1] = [
                        #    [df_tmp[cond].values, 
                        #     1-df_tmp[cond].values]]
                    else:
                        if (soft):
                            val = self.df_table.loc[
                                (ctx[i], pos[i], )].mean() 
                        else:
                            val = 0.5

            except:

                val = 0.5

            # Decimals conversion
             
            if (DECIMALS is not None):
                pred_proba[i:i+1] = [
                        [np.around(val, decimals=DECIMALS), 
                         np.around(1-val, decimals=DECIMALS)]]
            else:
                pred_proba[i:i+1] = [
                        [val, 
                         1-val]]
           
                    # Ou utiliser le MAX ?
        
        return pred_proba



def compression_packet_parallel(
    prob,
    packet, 
    packet_target,
    alphabet_size, 
    timesteps,
    level_compress):

    # We can't know the compression level if 
    # value is below the overhead
    # level_compress count from 0 !
    #assert (min(level_compress) > timesteps-1) and \
    #       (max(level_compress) < packet.shape[0]+timesteps-1)
    level_compress_array = np.array(
            level_compress)
    idx_level_compress = np.where(
        level_compress_array < (packet.shape[0] + \
            timesteps-1))[0]
    #print("[DEBUG][compression_packet_parallel] idx_level_compress:", 
    #        idx_level_compress)
    level_compress = level_compress_array[
            idx_level_compress]

    enc = ArithmeticEncoder(
        32, bitout=None, write_mode=False)
    prob_tmp = np.ones(alphabet_size)/alphabet_size
    cumul = np.zeros(
        alphabet_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(
        prob_tmp*10000000 + 1) 
    
    # Init with uniform law
    for j in range(timesteps):
        init_val = packet[
            0, j].astype(np.uint8) # packet[0, j]
        enc.write(cumul, init_val)

    lengths_level_compress = []
    idx_level_compress = 0

    #print("[DEBUG][compression_packet_parallel] packet.shape: ", packet.shape)
        
    for i in range(
        0, packet.shape[0]):
        #print("[DEBUG][compression] prob[i] : ", prob[i])

        prob_tmp = prob[i:i+1]
        y_original = int(
            np.argmax(packet_target[i:i+1]))

        cumul[1:] = np.cumsum(prob_tmp*10000000 + 1)
        enc.write(cumul, y_original)

        # Eval compression at certain level
        if ((i+timesteps-1) == level_compress[
                idx_level_compress]):

            length_level_compress = len(
                enc.data_compress)
            lengths_level_compress \
                .append(length_level_compress)
 
            if (idx_level_compress < len(level_compress)-1):
                #print("[DEBUG] idx_level_compress: ", 
                #        idx_level_compress)
                idx_level_compress += 1

    enc.finish()

    return enc.data_compress, lengths_level_compress


def get_params_compression(
    model,
    generator,
    headers_length,
    num_sin,
    look_back_context,
    look_back_packet,
    max_length,
    alphabet_size,
    range_list_IDs,
    range_list_IDs_tmp,
    level_compress,
    packets_rank,
    left_padding): 
    # range_list_IDs.shape[0]
    # give number of packet
    
    idx_list_IDs = np.where(
        range_list_IDs_tmp[0, 0] == \
        range_list_IDs[:, 0])[0].item()
    
    batchs = (headers_length*8) - look_back_packet
    idx_batch = int((np.cumsum(batchs[:idx_list_IDs+1])[-1] - \
                 batchs[idx_list_IDs]).item())
    params_compression = [None] * range_list_IDs_tmp.shape[0]
    
    # Extraction des packets
    # et compression
    for i in range(
        idx_list_IDs, 
        idx_list_IDs+range_list_IDs_tmp.shape[0]):
        
        start_idx, end_idx = \
            range_list_IDs[i, 0], range_list_IDs[i, 1]
        #print("[DEBUG][get_params_compression] range_list_IDs[i, 0]: ", range_list_IDs[i, 0])
        #print("[DEBUG][get_params_compression] range_list_IDs[i, 1]: ", range_list_IDs[i, 1])

        quantity = int(end_idx - start_idx) # Check if good coverage !
        #print("[DEBUG][get_params_compression] quantity: ", quantity)
        # quantity get the nb of block in packet
        
        
        header_length = int(
            headers_length[i]*8)
        pkt_rank = int(packets_rank[i])

        
        # Init array
        idx_tmp = 0 # ou j-idx_batch
        pkt_seq = np.zeros(
            (quantity, look_back_packet, 1))
        pkt_pos_seq = np.zeros(
            (quantity, look_back_packet, 1))
        
        if (left_padding):
            ctx_seq = np.zeros(
                (quantity, look_back_context, 
                 max_length-look_back_packet))
        else:
            ctx_seq = np.zeros(
                (quantity, look_back_context, max_length))

        pkt_target = np.zeros(
            (quantity, alphabet_size))
        
        for j in range(
            start_idx, end_idx): # idx_batch, idx_batch+quantity): # Verifier le end_idx (inclue ou non ?)

            # Get array
            ctx_seq_tmp, pkt_sin_seq_tmp, \
                pkt_target_tmp = generator \
                    .__getitem__(index=j)
            
            # Set array to packet's array
            pkt_seq[idx_tmp:idx_tmp+1] = pkt_sin_seq_tmp[:, :, 0:1]
            pkt_pos_seq[idx_tmp:idx_tmp+1] = pkt_sin_seq_tmp[:, :, -1:]
            pkt_target[idx_tmp:idx_tmp+1] = pkt_target_tmp
            ctx_seq[idx_tmp:idx_tmp+1] = ctx_seq_tmp
            idx_tmp += 1
            
        idx_batch += quantity

        # Reshape
        ctx_seq_reshape = ctx_seq.reshape(
            ctx_seq.shape[0], -1)
        pkt_seq_reshape = pkt_seq[:, :, 0]
        #print("[DEBUG][get_params_compression] ctx_seq_reshape.shape: ", ctx_seq_reshape.shape)
        #print("[DEBUG][get_params_compression] pkt_seq_reshape.shape: ", pkt_seq_reshape.shape)

        # Extract input data
        #print("[DEBUG] ctx_seq.shape: ", ctx_seq.shape)
        #print("[DEBUG] pkt_seq.shape: ", pkt_seq.shape)
        #print("[DEBUG] ctx_seq_reshape.shape: ", 
        #        ctx_seq_reshape.shape)
        #print("[DEBUG] pkt_seq_reshape.shape: ", 
        #        pkt_seq_reshape.shape)
        data_input = np.concatenate(
            (ctx_seq_reshape, pkt_seq_reshape), axis=-1)
        #print("[DEBUG][get_params_compression] data_input.shape: ", data_input.shape)
        if (pkt_rank >= look_back_context):
            pkt_ranks = [look_back_context] * \
                int(quantity)
        else:
            pkt_ranks = [pkt_rank] * \
                int(quantity)
        
        # Get proba
        prob = model.predict(
                 data=data_input, 
                 ctx=pkt_ranks, 
                 pos=pkt_pos_seq[:, 0, 0]\
                        .astype(int), 
                 soft=SOFT_MODE)

        # Array for params compression
        params_compression[i-idx_list_IDs] = (prob,
                                 pkt_seq,
                                 pkt_target,
                                 alphabet_size,
                                 look_back_packet,
                                 level_compress)
        
    return params_compression


def compression(model,
              packets,
              packets_rank,
              generator, 
              headers_length,
              max_length,
              range_list_IDs,
              look_back_context,
              look_back_packet,
              alphabet_size,
              num_sin,
              batch_packet,
              level_compress,
              left_padding):
    
    # Define data for results
    df_results = pd.DataFrame()
    
    # Pre fill list for parallel computing    
    size_compress = [None] * headers_length.shape[0]
    size_level_compress = np.empty(
        (headers_length.shape[0], 
         len(level_compress)))
    
    for b in range(
        0, headers_length.shape[0], 
        batch_packet): #pkt_sin_seq.shape[0]):
        
        # On prépare les paquets qui vont être compressé !
        index_start = b
        index_end = min(b+batch_packet, 
                        headers_length.shape[0])
            
        # We get the parameters for compression
        range_list_IDs_tmp = range_list_IDs[
            index_start:index_end]
        params_compression = get_params_compression(
                model=model,
                range_list_IDs=range_list_IDs, 
                range_list_IDs_tmp=range_list_IDs_tmp,
                generator=generator,
                headers_length=headers_length,
                num_sin=num_sin,
                look_back_context=look_back_context,
                look_back_packet=look_back_packet,
                max_length=max_length,
                alphabet_size=alphabet_size,
                level_compress=level_compress,
                packets_rank=packets_rank,
                left_padding=left_padding)

        # Parallel compression
        
        with mp.Pool() as workers:
            #packets_compress, lengths_level_compress = \
            results = workers.starmap(
                    compression_packet_parallel, 
                    params_compression)

            #print("[DEBUG] results: ", results)
            #print("[DEBUG] results[0]: ", results[0])
            #print("[DEBUG] results[0][0]: ", results[0][0])
            #print("[DEBUG] results[0][1]: ", results[0][1])

            #packets_compress = results[0]

            #print("[DEBUG] len(results): ", 
            #        len(results))
            #print("[DEBUG] results: ", 
            #        results)

            packets_compress_size = [
                len(packet[0]) for packet in results]
            lengths_level_compress_tmp = [
                lengths_level[1] for lengths_level in results]

            lengths_level_compress = np.ones(
                    (len(lengths_level_compress_tmp), 
                     len(level_compress)))*(-1)

            # Iterate to set each list to an array
            for i in range(
                len(lengths_level_compress_tmp)):
                
                length_tmp = len(
                    lengths_level_compress_tmp[i])
                lengths_level_compress[
                    i, :length_tmp] = lengths_level_compress_tmp[i]

            #print("[DEBUG] lengths_level_compress.shape: ", 
            #        lengths_level_compress.shape)
            #print("[DEBUG] lengths_level_compress_tmp.shape: ", 
            #        lengths_level_compress_tmp.shape)

            #lengths_level_compress[:, 
            #        :len(lengths_level_compress_tmp[0])] = lengths_level_compress_tmp
        
        size_compress[
            index_start:index_end] = packets_compress_size 
        size_level_compress[
            index_start:index_end, :] = lengths_level_compress

    
    df_results['size_compress'] = size_compress
    df_results['headers_length'] = (headers_length*8).astype(int) # Transform to bit
    df_results['packets_rank'] = packets_rank

    # Set compression level
    columns = [f"level_compress_{i}" 
                for i in level_compress]
    #print("[DEBUG] size_level_compress.shape: ", size_level_compress.shape)
    #print("[DEBUG] len(columns): ", len(columns))
    df_results.at[:, columns] = size_level_compress \
                                    .astype(int)
    
    return df_results


class DataGeneratorContinuous(keras.utils.Sequence):
    """Generator to process data for learning.

    Args:
        keras (tf.keras.utils.Sequence): Keras Sequence object.
    """
    def __init__(self, 
                 list_IDs,
                 
                 look_back_context,
                 look_ahead_context,
                 look_back_packet,
                 look_ahead_packet,
                
                 packets_rank,
                 packets, 
                 headers_length,
                 max_length,
                 
                 indexes_packet,
                 indexes_block,
                 
                 left_padding=False,
                 batch_size=32,
                 num_sin=8,
                 alphabet_size=2, 
                 shuffle=True):
        """Constructor.

        Args:
            list_IDs (list): identifier of each data block (in the window size).
            look_back_context (int): context size.
            look_ahead_context (int): jump of the context.
            look_back_packet (int): window size.
            look_ahead_packet (int): jump of the window size.
            packets_rank (np.array): ranks of packets in their flow.
            packets (np.array): packets to process.
            headers_length (np.array): size of header for each packet.
            max_length (int): maximum length of header to process.
            indexes_packet (np.array): indexes of packets.
            indexes_block (np.array): indexes of blocks.
            left_padding (bool, optional): apply padding to the packet to compress. Defaults to False.
            batch_size (int, optional): size of the batch. Defaults to 32.
            num_sin (int, optional): number of sinusoids used. Defaults to 8.
            alphabet_size (int, optional): number of different type of symbol in data. Defaults to 2.
            shuffle (bool, optional): shuffle the data or not. Defaults to True.
        """
        self.left_padding = left_padding
        self.batch_size = batch_size

        self.list_IDs = list_IDs # Index of block !
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Identification data
        #self.flows_id = flows_id
        self.packets_rank = packets_rank
        self.packets = packets
        self.headers_length = headers_length
        
        # Parameters data
        self.num_sin = num_sin
        # +2 because we add position information
        self.num_feats = self.num_sin + 2 
        self.alphabet_size = alphabet_size
        self.max_length = max_length
        
        self.look_back_context = look_back_context
        self.look_ahead_context = look_ahead_context
        self.look_back_packet = look_back_packet
        self.look_ahead_packet = look_ahead_packet
        
        # Generation of indexes
        nb_batch_block_min = (((self.headers_length*8) - 
                              self.look_back_packet) // self.batch_size) * self.batch_size
        nb_batch_block_max = ((self.headers_length*8) - 
                             self.look_back_packet)
        nb_batch_block = np.maximum(nb_batch_block_min, nb_batch_block_max)
        self.indexes_max_packet = np.cumsum(nb_batch_block)
        self.indexes_min_packet = np.concatenate((np.zeros(1),
                                                  self.indexes_max_packet[0:-1]))
        self.indexes_max_packet = self.indexes_max_packet - 1

        # Indexes packet and block pre computed !
        self.indexes_packet = indexes_packet
        self.indexes_block = indexes_block
        
        # Generate sinusoide
        self.sin = np.empty((self.max_length, 0), dtype=np.float64)
        self.pos = np.arange(0, self.max_length)
        for j in range(1, self.num_sin+1):
            sin_tmp = self.__gen_sin(
                F=j, Fs=self.max_length, phi=0, A=1, 
                mean=0.5, period=j, center=0.5).reshape((self.max_length, 1))
            self.sin = np.concatenate((self.sin, sin_tmp), axis=-1)    
        self.sin_seq = create_windows(
            self.sin, window_shape=self.look_back_packet, 
            end_id=-self.look_ahead_packet)
        self.pos_seq = create_windows(
            self.pos, window_shape=self.look_back_packet, 
            end_id=-self.look_ahead_packet)
        
        
    def __gen_sin(self, F, Fs, phi, A, mean, period, center):
        """Generate sinusoids.

        Args:
            F (int): frequency.
            Fs (int): sampling frequency.
            phi (int): phase.
            A (int): amplitude.
            mean (int): moyenne (obsolete to remove).
            period (int): period.
            center (int): mean of the signal.

        Returns:
            np.array: signal generated.
        """
        T = (1/F)*period # Period if 10/F = 10 cycles
        Ts = 1./Fs # Period sampling
        N = int(T/Ts) # Number of samples
        t = np.linspace(0, T, N)
        signal = A*np.sin(2*np.pi*F*t + phi) + center
        return signal

    def __len__(self):
        """Denotes the number of batches per epoch.

        Returns:
            int: number of batchs.
        """
        return int(np.floor(
            (self.list_IDs.size /  self.batch_size)))
    

    def __getitem__(
        self, index):
        """Generate one batch of data.

        Args:
            index (int): index of data to get.

        Returns:
            list: context, packet en bit to predict. 
        """
        
        # Get index of block
        index_start = self.indexes[
            index] * self.batch_size
        index_end = index_start + self.batch_size
        
        # Get indexes packet and block
        indexes_packet = self.indexes_packet[
            index_start:index_end]
        indexes_block = self.indexes_block[
            index_start:index_end]
        
        # Get the data
        ctx, pkt, y = self.__data_generation(
                indexes_packet, indexes_block)

        if (self.left_padding):
            return [ctx[:, :, self.look_back_packet:], pkt, y]
        else:
            return [ctx, pkt, y]


    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(
            np.floor(len(self.list_IDs)/self.batch_size)) \
            .astype(int)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __get_context(
        self, indexes_packet):
        """Get context associated to packets.

        Args:
            indexes_packet (np.array): array of packets indexes.

        Returns:
            np.array: array of context.
        """
        
        ctx = np.zeros(
            (indexes_packet.size, 
             self.look_back_context, 
             self.max_length))
        
        for i in range(ctx.shape[0]):
            rank = indexes_packet[i] #list_IDs[i]
            
            pad_size = max(0, self.look_back_context - \
                self.packets_rank[rank])
            
            pkt_size = self.look_back_context - \
                pad_size
            
            ctx_tmp = np.concatenate(
                    (np.zeros((pad_size, self.max_length)), 
                     self.packets[rank-pkt_size:rank]), axis=0)
            
            ctx_tmp = ctx_tmp.reshape(
                1, self.look_back_context, -1)
                
            ctx[i] = ctx_tmp
        
        # Repeat context based on header length
        nbs_repeat = ((self.headers_length[indexes_packet]*8) - \
              self.look_back_packet).astype(int)
        ctx_seq = np.repeat(ctx, nbs_repeat, axis=0)
        
        return ctx_seq

    
    def __get_packet(self, 
                     indexes_packet, 
                     indexes_block):
        """Get block after applying sliding window in packet 
        and his associated target. The sinusoids is also 
        concatenated.

        Args:
            indexes_packet (np.array): indexes of packets.
            indexes_block (np.array): indexes of blocks.

        Returns:
            tuple: blocks of sliding window with sinusoids and the targets associated.
        """
        
        #nb_repeat = ((self.headers_length[indexes_packet]*8) - \
        #      self.look_back_packet).sum().astype(int) #[0]
        nb_repeat = indexes_block.size
        
        pkt_seq = np.zeros(
            (nb_repeat, self.look_back_packet, 
             self.num_feats))
        
        y_seq = np.zeros(
            (nb_repeat, self.alphabet_size))
        
        index_start = int(0)
        for idx_pkt, idx_block in zip(
            indexes_packet, indexes_block):
            
            idx_start = idx_block
            idx_end = idx_block + self.look_back_packet
            
            # Set packet
            pkt = self.packets[
                idx_pkt, :int(self.headers_length[idx_pkt]*8)]
            
            pkt_seq_tmp = pkt[idx_start:idx_end] \
                .reshape(-1, self.look_back_packet, 1)
            
            y_seq_tmp = pkt[idx_end]
            y_seq_tmp = tf.keras.utils.to_categorical(
                y=y_seq_tmp.ravel(), 
                num_classes=self.alphabet_size, 
                dtype='float64')
            
            pkt_seq[index_start:index_start+1, :, 0:1] = pkt_seq_tmp
            pkt_seq[index_start:index_start+1, :, 1:-1] = self.sin_seq[
                idx_start:idx_start+1]
            pkt_seq[index_start:index_start+1, :, -1:] = self.pos_seq[
                idx_start:idx_start+1]
            y_seq[index_start:index_start+1] = y_seq_tmp
            
            index_start += 1
            
        return pkt_seq, y_seq

    
    def __data_generation(
        self, indexes_packet, indexes_block):
        """Generates data containing batch_size samples.

        Args:
            indexes_packet (np.array): indexes of packets.
            indexes_block (np.array): indexes of blocks.

        Returns:
            np.array: context, packet en bit to predict.
        """  
        # Initialization
        ctx = self.__get_context(
            indexes_packet)
        
        pkt, y = self.__get_packet(
            indexes_packet, indexes_block)
        
        # Correct context shape
        ctx = ctx[:pkt.shape[0]]
        
        # We take random blocks among the packages, 
        # not necessarily the selected blocks...
        # Less clean but simpler...
        indexes = np.arange(0, pkt.shape[0])\
                    .astype(int)
        
        # Shuffle to avoid to always keep the last block non learned !
        ctx = ctx[indexes]
        pkt = pkt[indexes]
        y = y[indexes]
        
        return ctx, pkt, y




#############################################
# LAUNCH TRAINING
#############################################


# PREPARE DATA




start_time = datetime.datetime.now()



# For checksum AND combinaison !
# From : https://www.thegeekstuff.com/2012/05/ip-header-checksum/

def sum_checksum(
    array_bit_a, array_bit_b):
    
    # Array must have the same length
    assert len(array_bit_a) == len(array_bit_b)
    
    # Sum of array
    array_bit_sum = np.zeros(
        (len(array_bit_a)), dtype=int)
    
    # For addition
    carry = 0
    
    # For each bit
    for i in reversed(range(len(array_bit_a))):
        
        # Get value
        value_a = array_bit_a[i]
        value_b = array_bit_b[i]
        sum_value = value_a + value_b + carry
        
        # Apply sum
        if (sum_value == 0):
            carry = 0
            array_bit_sum[i] = 0
        elif (sum_value == 1):
            carry = 0
            array_bit_sum[i] = 1
        elif (sum_value == 2):
            carry = 1
            array_bit_sum[i] = 0
        elif (sum_value == 3):
            carry = 1
            array_bit_sum[i] = 1
            
    # If the result is above origianl length
    if (carry):
        array_bit_tmp = np.zeros(
            (len(array_bit_a)), dtype=int)
        array_bit_tmp[-1] = 1
        array_bit_sum = sum_checksum(
            array_bit_sum, array_bit_tmp)
            
    return array_bit_sum
    

def complement_chekcsum(array):
    """Compute the complement.

    Args:
        array (np.array): array of bit.

    Returns:
        np.array: Complement of the array.
    """
    array_complement = \
        (array - 1)*(-1)
    return array_complement

def checksum(array, size=8):
    """Compute the checksum.

    Args:
        array (np.array): array of bit.
        size (int, optional): size of the checksum 
        field. Defaults to 8.

    Returns:
        np.array: array of bit.
    """

    # Check array size is above 
    # checksum field size
    assert array.size >= size
    
    # Reshape to good size
    multiple = array.size / size
    
    # multiple == 1 therefore array.size = size
    # So we add with 0 we need to pad
    if ((array.size % size == 0) and 
        (multiple != 1)):
        array_reshape = \
            array.reshape((-1, size))
    else:
        
        # Compute coeff and get 
        # missing bit
        if (multiple == 1):
            coeff = 2
        else:
            coeff = np.ceil(array.size / size)
            
        # Compute nb_bits
        nb_bit = int(coeff*size - array.size)
        
        # Add zeros for padding
        array_tmp = np.concatenate(
            (array, np.zeros((nb_bit,)))) 

        # Reshape array
        array_reshape = \
            array_tmp.reshape((-1, size))
        
    
    # For each bytes
    for i in range(
        0, array_reshape.shape[0], 2):
        
        # Get first value
        if (i == 0):
            array_bit_a = array_reshape[i]
            array_bit_b = array_reshape[i+1]
        else:
            array_bit_a = array_bit_sum
            array_bit_b = array_reshape[i]
        
        # Apply sum
        array_bit_sum = sum_checksum(
            array_bit_a=array_bit_a, 
            array_bit_b=array_bit_b)
        
    # Apply complement
    array_bit_checksum = \
        complement_chekcsum(
            array=array_bit_sum)
        
    return array_bit_checksum



## int_to_bin
def int_to_bin(x, size=8):
    val = str(bin(x))[2:]
    s = len(val)
    gap = size - s
    
    if (gap < 0):
        raise "Error size limit too short"
        #size = max(s, size)
    
    gap_val = '0'*gap
    return gap_val+val



# Generate with another seed ! (for evaluation)
# "random", "checksum", "fixed", "inversion", "counter"

if (MODE_DATASET == "random"):

    # Reset the seed
    np.random.seed(43)

    arr_raw = \
        np.random.randint(
            0, 2, size=(NB_ROWS, NB_BITS))


elif ("checksum" in MODE_DATASET):

    # Reset the seed
    np.random.seed(43)


    if (MODE_DATASET == "checksum34"):
        # Checksum over 3 bytes
        CHECKSUM_LENGTH = 8
        
        # Create array of int
        arr_raw = np.random.randint(
            0, 2, 
            size=(NB_ROWS, int(NB_BITS*(3/4))),
            dtype=int)
        
    else:
        # Checksum classic over 16 bits
        CHECKSUM_LENGTH = 16
        
        # Create array of int
        arr_raw = np.random.randint(
            0, 2, 
            size=(NB_ROWS, int(NB_BITS*(2/4))),
            dtype=int)


    # Compute checksum
    arr_checksums = np.apply_along_axis(
        checksum, axis=1, 
        size=CHECKSUM_LENGTH,
        arr=arr_raw)
    arr_checksums = arr_checksums.astype(int)

    # Concatenate array
    arr_raw = np.concatenate(
        (arr_raw, arr_checksums), 
        axis=1)
    arr_raw = arr_raw.astype(int)


elif (MODE_DATASET == "fixed"):

    arr_raw = \
        np.ones((NB_ROWS, NB_BITS))
    arr_raw = arr_raw * \
        np.random.randint(
            0, 2, size=(1, NB_BITS))


elif (MODE_DATASET == "inversion"):

    # Reset the seed
    np.random.seed(43)

    ## Define array
    arr_raw = \
        np.zeros((NB_ROWS, NB_BITS))

    ## For each flow define random values

    # For each flow
    for i in range(
        0, NB_ROWS, MAX_PACKET_RANK):
        
        arr_raw_tmp_a = \
            np.random.randint(
                0, 2, size=(1, int(NB_BITS/2)))
        arr_raw_tmp_b = \
            np.random.randint(
                0, 2, size=(1, int(NB_BITS/2)))
        
        for j in range(MAX_PACKET_RANK):
            
            # Invert for next round
            if (j % 2 == 0):
                arr_raw_tmp = np.concatenate(
                    (arr_raw_tmp_b, arr_raw_tmp_a), 
                    axis=1)
            else:
                arr_raw_tmp = np.concatenate(
                        (arr_raw_tmp_a, arr_raw_tmp_b), 
                        axis=1)
            
            # Set to array
            if (i+j < NB_ROWS):
                arr_raw[i+j, :] = arr_raw_tmp


elif (MODE_DATASET == "counter"):

    # Reset the seed
    np.random.seed(43)
    
    ## int_to_bin
    def int_to_bin(x, size=8):
        val = str(bin(x))[2:]
        s = len(val)
        gap = size - s
        
        if (gap < 0):
            raise "Error size limit too short"
            #size = max(s, size)
        
        gap_val = '0'*gap
        return gap_val+val


    ## Define array
    arr_raw = \
        np.zeros((NB_ROWS, NB_BITS))


    # For each flow
    for i in range(
        0, NB_ROWS, 
        MAX_PACKET_RANK):
        
        # Generate number
        value = \
            np.random.randint(
                0, (2**NB_BITS)-MAX_PACKET_RANK, 
                size=(1,))[0]
        
        # For each flow rank
        for j in range(
            MAX_PACKET_RANK):
        
            ## Convert number to binary
            value_bin = int_to_bin(
                value, size=NB_BITS)
        
            ## Create array from binary
            ## values
            value_bin_array = \
                [int(bit) for bit in value_bin]
            
            # Set to array
            if (i+j < NB_ROWS):
                arr_raw[i+j, :] = value_bin_array
            
            # Add + 1
            value = value + 1


elif (MODE_DATASET == "fixed_flow"):

    # Reset the seed
    np.random.seed(43)

    ## Define array
    arr_raw = \
        np.zeros((NB_ROWS, NB_BITS))


    ## For each flow define random values

    # For each flow
    for i in range(
        0, NB_ROWS, MAX_PACKET_RANK):

        arr_raw_tmp = \
            np.random.randint(
                0, 2, size=(1, NB_BITS))

        for j in range(MAX_PACKET_RANK):

            # Set to array
            if (i+j < NB_ROWS):
                arr_raw[i+j, :] = arr_raw_tmp


elif (MODE_DATASET == "combinaison"):

    # Checksum over 3 bytes
    CHECKSUM_LENGTH = 8

    ## Define array
    arr_raw = \
        np.zeros((NB_ROWS, NB_BITS))

    # Generate fixed values
    arr_raw_fixed_tmp = \
            np.random.randint(
                0, 2, size=(1, int(NB_BITS/8)))

    # Reset the seed AFTER SETTING
    # fix values !
    np.random.seed(43)

    ## For each flow define random values

    # For each flow
    for i in range(
        0, NB_ROWS, MAX_PACKET_RANK):

        # Generate number for compteur
        value = \
            np.random.randint(
                0, (2**int(NB_BITS/4))-MAX_PACKET_RANK, 
                size=(1,))[0]

        #arr_raw_cmp_tmp = \
        #    np.random.randint(
        #        0, 2, size=(1, int(NB_BITS/4)))

        ## Generate fixed flow
        arr_raw_fixed_flow_tmp = \
            np.random.randint(
                0, 2, size=(1, int(NB_BITS/8)))

        for j in range(MAX_PACKET_RANK):

            ## Convert number to binary
            value_bin = int_to_bin(
                    value, size=int(NB_BITS/4))

            ## Create array from binary
            ## values
            value_bin_array = \
                [int(bit) for bit in value_bin]
            value_bin_array = \
                np.array(value_bin_array)\
                .reshape(-1, int(NB_BITS/4))

            # Set to array
            #if (i+j < NB_ROWS):
            #    arr_raw[i+j, :] = value_bin_array        

            # Generate random part
            arr_raw_random_tmp = \
                np.random.randint(
                    0, 2, size=(1, int(NB_BITS/4)))

            # Generate fixed et fixed flow

            ## Concat fixed bytes
            arr_raw_fixed_all_tmp = \
                np.concatenate(
                    (arr_raw_fixed_tmp, 
                     arr_raw_fixed_flow_tmp), 
                    axis=1)

            # Invert for next round
            if (j % 2 == 0):

                # Apply concatenation
                arr_raw_tmp = np.concatenate(
                    (arr_raw_random_tmp, 
                     arr_raw_fixed_all_tmp,
                     value_bin_array, #arr_raw_cmp_tmp
                    ), 
                    axis=1)

            else:

                # Apply concatenation
                arr_raw_tmp = np.concatenate(
                    (value_bin_array, #arr_raw_cmp_tmp,
                     arr_raw_fixed_all_tmp,
                     arr_raw_random_tmp), 
                    axis=1)

            #print("[DEBUG] arr_raw_tmp.shape: ", 
            #          arr_raw_tmp.shape)

            # Compute checkusm
            arr_raw_ckecksum_tmp = \
                np.apply_along_axis(
                    checksum, axis=1, 
                    size=CHECKSUM_LENGTH,
                    arr=arr_raw_tmp)

            #print("[DEBUG] arr_raw_ckecksum_tmp.shape: ", 
            #          arr_raw_ckecksum_tmp.shape)

            # Concat all bytes
            arr_raw_tmp = np.concatenate(
                (arr_raw_tmp, 
                 arr_raw_ckecksum_tmp), 
                axis=1)

            # Set to array
            if (i+j < NB_ROWS):
                arr_raw[i+j, :] = arr_raw_tmp

            # Add + 1
            value = value + 1

    # Convert to int
    arr_raw = arr_raw.astype(int)



# Print two lines to see similarity or not 
print("[DEBUG] arr_raw[:3]: ", arr_raw[:3])
print("[DEBUG] arr_raw.shape: ", arr_raw.shape)


# Add padding
if (LEFT_PADDING):

    arr_raw_padding = \
        np.zeros((NB_ROWS, LOOK_BACK_PACKET))

    arr_raw = np.concatenate(
        (arr_raw_padding, arr_raw), 
        axis=1)




# DEFINE VARIABLES FOR GENERATOR




## Set packet_rank
packets_rank_tmp = np.arange(0, MAX_PACKET_RANK)
packets_rank = \
    np.repeat([packets_rank_tmp], 
              int(NB_ROWS/MAX_PACKET_RANK), axis=0).ravel()

## Set packets
packets = arr_raw

## Set headers 
if (LEFT_PADDING):
    headers_length = np.repeat(
        [(NB_BITS+LOOK_BACK_PACKET) / 8], 
        NB_ROWS, 
        axis=0)
else:
    headers_length = np.repeat(
        [NB_BITS/8], 
        NB_ROWS, 
        axis=0)

## Set max length
max_length = packets.shape[-1]

# Define block length
if (LEFT_PADDING):
    block_length = np.repeat(
        [NB_BITS], 
        NB_ROWS, axis=0)
else:
    block_length = np.repeat(
        [NB_BITS-LOOK_BACK_PACKET], 
        NB_ROWS, axis=0)

# Get the list of block
cumsum_block = np.cumsum(
    block_length)
max_block = cumsum_block.max()

# Set all indexes
list_IDs = np.arange(
    max_block, dtype=int)
indexes_packet = np.repeat(
    np.arange(0, block_length.size, dtype=int), 
    block_length, axis=0)

# Valuer du début de chaque bloc
#  de chaque paquet
cumsum_block_tmp = np.zeros(
    cumsum_block.size, dtype=int)
cumsum_block_tmp[1:] = cumsum_block[:-1]

indexes_block = np.repeat(
    cumsum_block_tmp, 
    block_length, axis=0)
indexes_block = list_IDs - indexes_block





# EXTRACT LEVEL COMPRESS




# Set compression level
LEVEL_COMPRESS = []
for i in range(0, NB_BITS, 1):
    LEVEL_COMPRESS.append(i)





# WE TAKE NEW LIST ID WITH UNIQUE ELEMENT !




print("[DEBUG] BEFORE list_IDs.shape: ", list_IDs.shape)
print("[DEBUG] BEFORE indexes_packet.shape: ", indexes_packet.shape)
print("[DEBUG] BEFORE indexes_block.shape: ", indexes_block.shape)


# Get index for min and max element
# idx_unique is ordonned !
max_range = cumsum_block
min_range = np.concatenate(
    ([0], cumsum_block[:-1]))


index_start = 0
index_end = 0

range_list_IDs = np.zeros(
    (list_IDs.shape[0], 2))
range_index_start = 0
range_index_end = 0


for idx_start, idx_end in zip(
    min_range, max_range):
    
    # Update index
    range_index_end = range_index_start + 1    
    range_list_IDs[range_index_start:range_index_end] = [
        list_IDs[idx_start], list_IDs[idx_end-1]+1] # [0, +inf[
    range_index_start = range_index_end
    
    # Update index
    block_size = (idx_end - idx_start)
    #index_end = index_start + block_size

    #index_start = index_end

range_list_IDs = \
    range_list_IDs[:range_index_end].astype(int)





# CREATE GENERATOR





# On shuffle pour etre sur que les adresses sont toutes couvertes
# étape de CHECK COVERAGE ne devitn plus nécessaire
list_IDs_test = list_IDs
indexes_packet_test = indexes_packet
indexes_block_test = indexes_block


# Set parameters
params = {'look_back_context': LOOK_BACK_CONTEXT,
          'look_ahead_context': LOOK_AHEAD_CONTEXT, # Par default on ne sais qu'avec 1
          'look_back_packet': LOOK_BACK_PACKET,
          'look_ahead_packet': LOOK_AHEAD_PACKET, # Par default on ne sais qu'avec 1
         
          'packets_rank': packets_rank,
          'packets': packets,
          'headers_length': headers_length,
          'max_length': max_length,
         
          'left_padding': LEFT_PADDING,
          'batch_size': BATCH_SIZE,
          'num_sin': NUM_SIN,
          'alphabet_size': ALPHABET_SIZE, 
          'shuffle': SHUFFLE}


# Generators
# Index = index de bloc de batch !
# Example:
# Si len(listID) = 264
# alors index MAX = 132 si BATCH_SIZE = 2
generator_test = DataGeneratorContinuous(
    list_IDs=list_IDs_test, 
    indexes_packet=indexes_packet_test,
    indexes_block=indexes_block_test,
    **params)





# LOAD MODEL 



    

# Name
if (CHECKSUM):
    if (CUSTOM_SIZE is not None):
        EXT_NAME_MODEL = f"_WITH_CHECKSUM_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}"
    else:
        EXT_NAME_MODEL = f"_WITH_CHECKSUM_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}"

else:
    if (CUSTOM_SIZE is not None):
        EXT_NAME_MODEL = f"_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}"
    else:
        EXT_NAME_MODEL = f"_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}"

if (KEEP_ERROR):
    EXT_NAME_MODEL += f"_KEEP_ERROR_{CUT_VALUE}"

# If the SELECTIVE mode is activated 
if (SELECTIVE):
    EXT_NAME_MODEL = EXT_NAME_MODEL + "_SELECTIVE"

# If the OPTIMAL mode is activated 
if (OPTIMAL):
    EXT_NAME_MODEL = EXT_NAME_MODEL + "_OPTIMAL"

# If the LEFT_PADDING mode is activated 
if (LEFT_PADDING):
    EXT_NAME_MODEL = EXT_NAME_MODEL + "_LEFT_PADDING"



df_huffman_groupby = pd.read_csv(f"{HUFFMAN_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME_MODEL}.csv", dtype={"key": str})
array_index_pos = np.load(f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME_MODEL}.npy")


# CHANGE DATAFRAME FORMAT 
def my_func(x, max_length=8):
    value = str(x)
    length = len(value)
    value_extend = "0"*(max_length-length)
    value = value_extend + value
    return value 


df_huffman_groupby['key'] = df_huffman_groupby['key'].map(
    lambda x : my_func(x, max_length=CUT_VALUE))
df_huffman_groupby = df_huffman_groupby.groupby(
    ['ctx', 'pos', 'key']).mean()


table_model = TableModel(df_table=df_huffman_groupby,
                         index_pos=array_index_pos.astype(np.int32),
                         alphabet_size=ALPHABET_SIZE)



# START COMPRESSION




gc.collect()
df_results = compression(model=table_model,
                         batch_packet=BATCH_PACKET,
                         generator=generator_test, 
                         packets=packets, 
                         packets_rank=packets_rank,
                         headers_length=headers_length, 
                         max_length=max_length,
                         range_list_IDs=range_list_IDs,
                         look_back_context=LOOK_BACK_CONTEXT,
                         look_back_packet=LOOK_BACK_PACKET,
                         num_sin=NUM_SIN,
                         alphabet_size=ALPHABET_SIZE,
                         level_compress=LEVEL_COMPRESS,
                         left_padding=LEFT_PADDING)

#df_results['idx_unique'] = idx_unique_test





###############
# SAVE RESULT
###############

df_results.to_csv(f"{RESULTS_DIR}df_{FULL_NAME}{EXT_NAME}_COMPRESSION.csv", index=False)

end_time = datetime.datetime.now()

print("TIME DIFF : ", end_time-start_time)
