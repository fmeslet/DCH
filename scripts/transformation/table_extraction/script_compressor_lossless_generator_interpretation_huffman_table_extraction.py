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

# Personnal functions
# import functions


#############################################
# SET PARAMETERS
#############################################


HUFFMAN_DIR = "MODELS/HUFFMAN/INTERPRETATION/"
HUFFMAN_PARTS_DIR = HUFFMAN_DIR + "PARTS/"
FIELDS_DIR = "RESULTS/FIELDS/INTERPRETATION/"
MODELS_DIR = "MODELS/INTERPRETATION/"
MAIN_DIR = "./DATA/"

# Set keras types
tf.keras.backend.set_floatx('float64')

# Context
LOOK_BACK_CONTEXT = 1 # On rajoute +1 car on prend le dernier paquet comme paquet à compresser...
LOOK_AHEAD_CONTEXT = 1 #TIMESTEPS
CONTEXT_SIZE = LOOK_BACK_CONTEXT # Nombre timesteps sur les contexts
CONTEXT_OUTPUT_SIZE = 30 # Size of contexte in model layer
# De preference < 128 car c'est la taille de la
# couche GRU avant

# Packet
LOOK_BACK_PACKET = 8
LOOK_AHEAD_PACKET = 1
NUM_SIN = 8
NUM_FEATS = NUM_SIN + 1

# Generator
SHUFFLE = False #True
ALPHABET_SIZE = 2
BATCH_SIZE = 1 #512 #2048

# Size added to header_length IN BYTES
EXTRA_SIZE = 0
CUSTOM_SIZE = 100 # Take the lead if extra size is define
CHECKSUM = True # True = Keep Checksum

# For huffman creation
CUT_VALUE = 20 # For array of index possition
KEEP_ERROR = True # Change EXT_NAME, remove KEEP ERROR !
OPTIMAL = False
SELECTIVE = True # if True he most rank bit is used (only one)
MAX_RANK = 29 # Max rank level explore with num_rank (not starting from 0)

# Generated dataset parameters
MAX_PACKET_RANK = 3 # Max rank support for packet
NB_BITS = 16*2 # Multiple for field inversion
NB_ROWS = 4000*MAX_PACKET_RANK # Multiple for the flow id NB_ROWS = x * MAX_PACKET_RANK
MODE_DATASET = "random" # "random", "checksum", "checksum34", "fixed", "inversion", "counter", "fixed_flow", "combinaison"
LEFT_PADDING = True # Padding dataset


# Name
# min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE) if CONTEXT_OUTPUT_SIZE == 0
# else LOOK_BACK_CONTEXT == 1 or 2 or 3 or 4...etc
# easy to set LOOK_BACK_CONTEXT == 0
FULL_NAME = f"LOSSLESS_CONTEXT{min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE)}_PACKET{LOOK_BACK_PACKET}_SIN{NUM_SIN}"


if (CHECKSUM):

    if (CUSTOM_SIZE is not None):
        EXT_NAME = f"_WITH_CHECKSUM_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}"
    else:
        EXT_NAME = f"_WITH_CHECKSUM_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}"

else:

    if (CUSTOM_SIZE is not None):
        EXT_NAME = f"_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_DATASET{MODE_DATASET}"
    else:
        EXT_NAME = f"_EXTRA_SIZE{EXTRA_SIZE}_MODE_DATASET{MODE_DATASET}"


print(f"MODEL : {FULL_NAME}{EXT_NAME}")

# Set the Seed for numpy random choice
np.random.seed(42)


#############################################
# USEFULL CLASS/FUNCTIONS
#############################################

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


class CompressorLossless(keras.Model):
    def __init__(self, 
                 encoder_context,
                 ed_lossless,
                 **kwargs):
        super(CompressorLossless, self).__init__(**kwargs)
        self.encoder_context = encoder_context
        self.ed_lossless = ed_lossless
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def call(self, data):
        
        inputs = data
        context = data[0]
        packet = data[1]
        target = data[2]
        packet = tf.cast(packet, tf.float64)
        
        x = self.encoder_context(context)
        
        x = tf.expand_dims(x, axis=1)
        x = tf.repeat(x, LOOK_BACK_PACKET, axis=1)
        
        x = tf.concat([x, packet], axis=-1)
        
        x = self.ed_lossless(x)

        loss_reconstruction = tf.reduce_sum(
            keras.losses.binary_crossentropy(target, x),
        )
        loss = loss_reconstruction
        
        self.loss_tracker.update_state(loss)
        
        return x

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            
        #print("[DEBUG][train_step] data : ", data)
        
        inputs = data
        context = data[0]
        packet = data[1]
        target = data[2]
        packet = tf.cast(packet, tf.float64)
        
        with tf.GradientTape() as tape:

            x = self.encoder_context(context)
            
            x = tf.expand_dims(x, axis=1)
            x = tf.repeat(x, LOOK_BACK_PACKET, axis=1) # shape_packet[1]
            
            x = tf.concat([x, packet], axis=-1)
            
            x = self.ed_lossless(x)
            
            loss_reconstruction = tf.reduce_sum(
                keras.losses.binary_crossentropy(target, x)
            )
            loss = loss_reconstruction

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.loss_tracker.update_state(loss)

        return {
            "tot": self.loss_tracker.result()
        }


def build_model_context_lossless(input_shape, output_size):
    
    encoder_inputs = keras.Input(shape=input_shape)
    
    x = encoder_inputs # ADDITION DE header shape et context
    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
    #        input_shape[-1], activation=tf.keras.layers.LeakyReLU(alpha=0.1)))(x)
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128,
                                activation='tanh', 
                                return_sequences=False, 
                                return_state=False))(x)
    #x = layers.Flatten()(x)
    x = layers.Dense(output_size, name="z", 
                     activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    encoder = keras.Model(encoder_inputs, x, name="encoder")
    
    return encoder


def build_model_lossless(
    input_shape, output_shape, input_dim):
    
    output_dim = int(output_shape**0.25)
    
    inputs = keras.Input(shape=input_shape) #_symbol
    inputs_result = inputs #.values
    
    #print("[DEBUG][build_model_lossless] inputs_result shape : ", tf.shape(inputs_result))
    
    inputs_symbol = tf.slice(inputs_result, [0, 0, 0], [-1, -1, 1])
    
    #print("[DEBUG][build_model_lossless] inputs_symbol shape : ", tf.shape(inputs_symbol))
    
    inputs_symbol_shape = [k for k in tf.shape(inputs_symbol)]
    inputs_symbol = tf.reshape(
        inputs_symbol, [inputs_symbol_shape[0], inputs_symbol_shape[1]])
    
    #print("[DEBUG][build_model_lossless] inputs_symbol shape : ", tf.shape(inputs_symbol))
    
    inputs_sin = tf.slice(inputs_result, [0, 0, 1], [-1, -1, -1])

    embed = tf.keras.layers.Embedding(input_dim=input_dim, #output_shape,
                                      output_dim=output_dim,
                                      embeddings_initializer="uniform",
                                      embeddings_regularizer=None,
                                      activity_regularizer=None,
                                      embeddings_constraint=None,
                                      mask_zero=False,
                                      input_length=input_shape[-1])
    embed_output = embed(inputs_symbol)
    embed_output_shape = [k for k in tf.shape(embed_output)]
    #print("embed_output_shape : ", embed_output_shape)
    
    x = tf.concat(
        [embed_output, inputs_sin], axis=-1)
    
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(32, #output_dim, #input_shape[-1],
                                activation='tanh', 
                                return_sequences=False, 
                                return_state=False))(x)
    x = layers.Dense(3, 
                     activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(output_shape, 
                     activation="softmax")(x)
    model = keras.Model(inputs, x, name="model")
    embedding_model = keras.Model(inputs, embed_output, name="embedding_model")
    
    return model, embedding_model


# GENERATOR AVEC LA NOTION
# de postion en retour

class DataGeneratorContinuous(keras.utils.Sequence):
    'Generates data for Keras'
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
        'Initialization'
        self.left_padding = left_padding
        self.batch_size = batch_size
        # Liste des index des packets (peut etre que les vals ou trains...)
        self.list_IDs = list_IDs # Index de block !
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
        
        # Generation des indexes
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
        T = (1/F)*period # Period if 10/F = 10 cycles
        Ts = 1./Fs # Period sampling
        N = int(T/Ts) # Number of samples
        t = np.linspace(0, T, N)
        signal = A*np.sin(2*np.pi*F*t + phi) + center
        return signal

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(
            (self.list_IDs.size /  self.batch_size)))
    
    # /!\ A adapter en fonction de la look_ahead_packet 
    # (par default on dit quel est a UN !)
    def __compute_batch_quantity(self, indexes_packet):
        #(self.headers_length[indexes_packet].sum() * 8)
        pkts_size = (self.headers_length[
            indexes_packet].sum() * 8)
        pkts_size_cut = pkts_size - \
            (indexes_packet.shape[0]*self.look_back_packet)
        return pkts_size_cut # / self.batch_size)
    

    def __getitem__(
        self, index):
        'Generate one batch of data'
        # On met à jour l'index avec le coté random !
        
        # Get index of block
        index_start = self.indexes[
            index] * self.batch_size
        index_end = index_start + self.batch_size
        
        # Get indexes packet and block
        indexes_packet = self.indexes_packet[
            index_start:index_end]
        indexes_block = self.indexes_block[
            index_start:index_end]
        
        # On prend le min et on ajoute !
        # On récupère les données
        ctx, pkt, y = self.__data_generation(
                indexes_packet, indexes_block)

        if (self.left_padding):
            return [ctx[:, :, self.look_back_packet:], pkt, y]
        else:
            return [ctx, pkt, y]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # On manipule des indexes de block !
        self.indexes = np.arange(
            np.floor(len(self.list_IDs)/self.batch_size)) \
            .astype(int)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
    def __get_context(
        self, indexes_packet):
        
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
        
        # Repeat le contexte en fonction de la header length
        nbs_repeat = ((self.headers_length[indexes_packet]*8) - \
              self.look_back_packet).astype(int)
        ctx_seq = np.repeat(ctx, nbs_repeat, axis=0)
        
        return ctx_seq

    
    def __get_packet(self, 
                     indexes_packet, 
                     indexes_block):
        # En fonction de la taille de batch size il faut 
        # penser à recupere un ou plusieurs paquets !
        
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
        'Generates data containing batch_size samples'      
        # Initialization
        ctx = self.__get_context(
            indexes_packet)
        
        pkt, y = self.__get_packet(
            indexes_packet, indexes_block) # On shuffle les packet et target
        
        # Correct context shape
        ctx = ctx[:pkt.shape[0]]
        
        # On prend des bloque alétoire parmis les paquet
        # pas forcement les blocks selectionné...
        # Moins propre mais + simple...
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



# PREPARE DATASETS



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
    array_complement = \
        (array - 1)*(-1)
    return array_complement

def checksum(array, size=8):
    
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

    arr_raw = \
        np.random.randint(
            0, 2, size=(NB_ROWS, NB_BITS))


elif ("checksum" in MODE_DATASET):

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



# Import df fields
if (LEFT_PADDING):
    df_fields = pd.read_csv(f"{FIELDS_DIR}df_FIELDS_{FULL_NAME}{EXT_NAME}_LEFT_PADDING.csv")
else:
    df_fields = pd.read_csv(f"{FIELDS_DIR}df_FIELDS_{FULL_NAME}{EXT_NAME}.csv")




# FILTER FIELDS DATA TO GET OPTIMAL
# get fields which explain 95% of variation in probabilities 
# ONLY IN OPTIMAL CASE !!!




if (OPTIMAL):
    
    def standardize(x, min_x, max_x, a, b):
      # x_new in [a, b]
        x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
        return x_new
    
    
    ## Normalize probability for each packet
    print("[DEBUG][if (OPTIMAL)] Start proba normalization")
    
    columns_proba = [f"proba_{i}" for i in range(29)]
    array = df_fields[columns_proba].values

    def my_func(x):
        return standardize(x=x, min_x=x.min(), 
                           max_x=x.max(), a=0, b=1)

    array_norm = np.apply_along_axis(
        my_func, 1, array)

    ## Array to remove field which
    ## not explain 95% of the variation !
    print("[DEBUG][if (OPTIMAL)] Start proba filtering")
    
    def my_func(x):
        idxs = np.where(x <= 0.05)
        x_new = np.ones(
            x.shape)
        x_new[idxs] = np.NaN
        return x_new

    array_filter = np.apply_along_axis(
        my_func, 1, array_norm)
    
    
    ## Array to remove field which
    ## not explain 95% of the variation !
    print("[DEBUG][if (OPTIMAL)] Start dataframe transformation")
    
    columns_rank = [f"rank_{i}" for i in range(29)]
    df_fields[columns_rank] = df_fields[columns_rank]*array_filter
    
    # Negative rank field implies no
    # values usefull !
    df_fields[columns_rank] = \
        df_fields[columns_rank].fillna(-1).astype(int)




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






# LOAD MODEL




# Change extension for model
# load
EXT_NAME_MODEL = EXT_NAME

if (LEFT_PADDING):
    EXT_NAME_MODEL += "_LEFT_PADDING" 


encoder_context = tf.keras.models.load_model(f"{MODELS_DIR}ENCODER_CONTEXT_{FULL_NAME}{EXT_NAME_MODEL}.h5", 
                                              custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
ed_lossless = tf.keras.models.load_model(f"{MODELS_DIR}ED_{FULL_NAME}{EXT_NAME_MODEL}.h5", 
                                          custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
embedder = tf.keras.models.load_model(f"{MODELS_DIR}EMBEDDER_{FULL_NAME}{EXT_NAME_MODEL}.h5", 
                                      custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
model = CompressorLossless(encoder_context=encoder_context, 
                           ed_lossless=ed_lossless)






# LOAD ARRAY WITH PROBA




# Create dict for relative position
# to flow
dict_index_index_rltv_flow = {}
for i in range(0, arr_raw.shape[0], 
               MAX_PACKET_RANK):
    for j in range(MAX_PACKET_RANK):
        dict_index_index_rltv_flow[i+j] = j
    
    
# Set relative position to flow
df_fields['index_rltv_flow'] = df_fields['index_packet'] \
            .map(dict_index_index_rltv_flow)
    
# Set index relative to packet
df_fields['index_rltv_packet'] = df_fields['index_block']





# CREATE GENERATOR





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

print("[DEBUG] df_fields AFTER : ", df_fields)
print("[DEBUG] df_fields['index_packet'] AFTER : ", df_fields['index_packet'])

# Set indexes array
df_fields_tmp = df_fields \
         .sort_values(by='list_IDs')

list_IDs_update_train = \
    df_fields_tmp['list_IDs'].values.astype(int)
indexes_packet_update_train = \
    df_fields_tmp['index_packet'].values.astype(int)
indexes_block_update_train = \
    df_fields_tmp['index_block'].values.astype(int)

'''print("[DEBUG] list_IDs_update_train_undersample.shape : ", list_IDs_update_train_undersample.shape)
print("[DEBUG] indexes_packet_update_train_undersample.shape : ", indexes_packet_update_train_undersample.shape)
print("[DEBUG] indexes_block_update_train_undersample.shape : ", indexes_block_update_train_undersample.shape)

print("[DEBUG] list_IDs_update_train_undersample : ", list_IDs_update_train_undersample)
print("[DEBUG] indexes_packet_update_train_undersample : ", indexes_packet_update_train_undersample)
print("[DEBUG] indexes_block_update_train_undersample : ", indexes_block_update_train_undersample)'''

# Generators
generator = DataGeneratorContinuous(
          list_IDs=list_IDs_update_train, 
          indexes_packet=indexes_packet_update_train,
          indexes_block=indexes_block_update_train,
          **params)





# EXTRACTION DE L'ARRAY DE POSITION




def get_unique_sorted(array,
                      return_counts=False,
                      axis=0):

    u, count = np.unique(
        array, 
        return_counts=True, 
        axis=axis)

    count_sort_ind = \
        np.argsort(-count)

    u_sorted = \
        u[count_sort_ind]
    count_sorted = \
        count[count_sort_ind]
        
    if (return_counts):
        return u_sorted, count_sorted
    else:
        return u_sorted



def get_unique_value_rank(
    df_ranks, 
    index_rltv_packet, 
    field_rank):
    
    cond = (df_ranks['index_rltv_packet'] == \
                index_rltv_packet)
    values = get_unique_sorted(df_ranks[
        cond][f'rank_{field_rank}'].values)
    #values = np.unique(df_ranks[
    #    cond][f'rank_{field_rank}'].values)
    return values


# Create array with position
# Pour chaque rang du paquet
# je connais les valeurs à regarder !


indexes_rltv_flow = df_fields[
    "index_rltv_flow"].value_counts().index.astype(int)
indexes_rltv_packet = df_fields[
    "index_rltv_packet"].value_counts().index.astype(int)
array_index_pos = np.zeros(
    (LOOK_BACK_CONTEXT+1,
     #indexes_rltv_packet.max()+1,
     indexes_rltv_packet.size, # No because indexes_rltv_packet maybe < max_length
     CUT_VALUE), dtype=np.int32)


# If mode is optimal non selected
# values are set to -1
if (OPTIMAL):
    array_index_pos = \
        array_index_pos - 1


for j in range(
    LOOK_BACK_CONTEXT+1):

    if (j == LOOK_BACK_CONTEXT):
        cond_rltv_flow = (
            df_fields["index_rltv_flow"] >= j)
    else:
        cond_rltv_flow = (
            df_fields["index_rltv_flow"] == j)

    df_fields_tmp = df_fields[cond_rltv_flow]
    indexes_rltv_packet = df_fields_tmp["index_rltv_packet"] \
                    .value_counts() \
                    .index.astype(int)
    
    for i, idx_rltv_packet in enumerate(
        indexes_rltv_packet):
       
        values = np.empty((0,))
        field_rank = 0
        value_quantity = 0

        while ((value_quantity < CUT_VALUE) and 
               field_rank < MAX_RANK): # of field_rank == max_fileds_rank (la on a tout parcouru..)

            # Aller chercher les valeurs
            values_tmp = get_unique_value_rank(
                df_fields_tmp, 
                index_rltv_packet=idx_rltv_packet,
                field_rank=field_rank)
            #values_tmp = values_tmp[
            #    ~np.isnan(values_tmp)]

            # If selective mode we select
            # the most popular value
            if (SELECTIVE):
                values_tmp = values_tmp[0:1]

            # Concatenate
            values_concat = np.concatenate(
                (values, values_tmp), axis=0)

            # Keep unique values
            values = np.unique(
                values_concat, axis=0)
            indexes = np.unique(
                values_concat, return_index=True)[1]
            values = np.array([values_concat[index] for index in sorted(indexes)])
            
            #print("[DEBUG] values: ", values)

            # Update field rank
            field_rank += 1

            # Update quantity
            value_quantity = values.size

        # Update values array
        values = values[:CUT_VALUE]

        # Only select values above 0 (case of OPTIMAL !!!)
        # -1 values can be present at the begginning
        values = values[values >= 0]

        if ((OPTIMAL or SELECTIVE) and 
            (values.size < CUT_VALUE)):
            # Pad values array with -1 to 
            # get CUT VALUES size
            values = np.lib.pad(values,
                            (0, CUT_VALUE-values.size),
                            'constant', constant_values=(-1))


        # Update value array
        array_index_pos[j:j+1, 
                        idx_rltv_packet:idx_rltv_packet+1] = \
                values.reshape(1, 1, -1)
        




# EXTRACTION HUFFMAN ARRAY





idx_df = 0
#max_val = CUT_VALUE * indexes.shape[0]
df_huffman = pd.DataFrame(
    columns=["key", "proba", 'pos', 'ctx']) 
df_huffman['proba'] = df_huffman[
    'proba'].astype(np.float32)


df_fields_sorted = df_fields \
    .sort_values(by='list_IDs') \
    .reset_index(drop=True)


for j in range(
    LOOK_BACK_CONTEXT+1):

    values = np.empty((0,))
    field_rank = 0
    value_quantity = 0

    if (j == LOOK_BACK_CONTEXT):
        cond_rltv_flow = (
            df_fields_sorted["index_rltv_flow"] >= j)
    else:
        cond_rltv_flow = (
            df_fields_sorted["index_rltv_flow"] == j)
        
    df_fields_flow = df_fields_sorted[cond_rltv_flow]
    indexes_rltv_packet = df_fields_flow["index_rltv_packet"] \
                     .value_counts() \
                     .index.values \
                     .astype(int)

    for i, idx_rltv_pkt in enumerate(
        indexes_rltv_packet):

        cond = (df_fields_flow[
            'index_rltv_packet'] == idx_rltv_pkt)
        df_fields_flow_tmp = \
               df_fields_flow[cond]

        # On récupère les indexes de getitems
        indexes_batch = df_fields_flow_tmp.index
        #indexes_batch = np.arange(
        #    0, df_fields_flow_tmp.shape[0], 
        #    dtype=int) #df_fields_flow_tmp['list_IDs'] \
                    #.values.astype(int)


        # On récupère les valeurs et la proba associées
        for k in indexes_batch:

            # Extract item
            ctx_seq, pkt_sin_seq, \
                y_seq = generator.__getitem__(index=k)
            pkt_sin_seq = pkt_sin_seq[:, :, :-1]

            # Reshape
            ctx_seq_ravel = ctx_seq.ravel().reshape(1, -1)
            pkt_seq_ravel = pkt_sin_seq[:, :, 0].ravel().reshape(1, -1)
            input_ravel = np.concatenate(
                (ctx_seq_ravel, pkt_seq_ravel), axis=-1)

            # Recuperation des rank a extraire
            indexes_rank = array_index_pos[j, 
                                           idx_rltv_pkt].copy()
            #print("[DEBUG] indexes_rank.shape: ", indexes_rank)

            if (OPTIMAL or SELECTIVE):
                # Filter indexes_rank == -1
                indexes_optimal = \
                    np.where(indexes_rank >= 0)[0]
                indexes_rank = indexes_rank[
                    indexes_optimal]

            # Extract values
            values_extract = input_ravel[
                :, indexes_rank] #indexes_tmp]

            # Create index
            values_extract = "".join(
                values_extract.ravel().astype(
                    np.uint8).astype(str))

            # Extract proba
            proba = model.predict([ctx_seq,
                                   pkt_sin_seq,
                                   np.zeros((pkt_sin_seq.shape[0], 2))], batch_size=1)

            #print("[DEBUG][for i][for k] proba : ", 
            #      proba)
            # Set to pandas
            df_huffman.loc[idx_df] = [values_extract, 
                                       proba.ravel()[0], 
                                       idx_rltv_pkt, j]
            idx_df += 1


        # Faire la sauvegarde de données temporaires
        
        ## Save data
        df_huffman_name = f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}"

        ### If keep error
        if (KEEP_ERROR):
            df_huffman_name = df_huffman_name + \
                f"_KEEP_ERROR_{CUT_VALUE}"
        else:
            df_huffman_name = df_huffman_name + \
                f"_{CUT_VALUE}"

        ### Selective
        if (SELECTIVE):
            df_huffman_name = df_huffman_name + \
                f"_SELECTIVE"

        ### Optimal
        if (OPTIMAL):
            df_huffman_name = df_huffman_name + \
                f"_OPTIMAL"

        # If padding on dataset
        if (LEFT_PADDING):
            df_huffman_name += f"_LEFT_PADDING"

        ### Set numbers
        df_huffman_name = df_huffman_name + \
                f"_{j}_{i}.csv"

        # Save DataFrame
        df_huffman.to_csv(df_huffman_name, index=False)
        # df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_OPTIMAL_{j}_{i}.csv", index=False)
 

        ## Reset dataframe
        df_huffman = pd.DataFrame(
                columns=["key", "proba", 'pos', 'ctx']) 
        df_huffman['proba'] = df_huffman[
            'proba'].astype(np.float32)




# CREATION DE L'ARRAY 



#df_huffman['key'] = df_huffman['key'].astype(str)
#df_huffman_groupby = df_huffman.groupby(
#    ['pos', 'key'])['proba'].mean()


#############################################
# SAVE DATAFRAME
#############################################


arr_name = f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}"

# If keep error
if (KEEP_ERROR):
    arr_name = arr_name + \
        f"_KEEP_ERROR_{CUT_VALUE}"
else:
    arr_name = arr_name + \
        f"_{CUT_VALUE}"

# Selective
if (SELECTIVE):
    arr_name = arr_name + \
        f"_SELECTIVE"

# Optimal
if (OPTIMAL):
    arr_name = arr_name + \
        f"_OPTIMAL"

# If padding on dataset
if (LEFT_PADDING):
    arr_name += f"_LEFT_PADDING"

# Add extension
arr_name += f".npy"

# Save array
np.save(arr=array_index_pos, file=arr_name)
# np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_OPTIMAL.npy")
