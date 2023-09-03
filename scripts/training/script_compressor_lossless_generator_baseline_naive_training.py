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
from sklearn import preprocessing

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Personnal functions
# import functions


#############################################
# SET PARAMETERS
#############################################


BASELINE_DIR = "MODELS/BASELINE_NAIVE/"
BASELINE_PARTS_DIR = BASELINE_DIR + "PARTS/"
RESULTS_DIR = "RESULTS/"
MODELS_DIR = "MODELS/"
MAIN_DIR = "./DATA/"

PROTO = "MQTT_IEEE"


# USELESS BUT KEEP FOR THE GENERATOR !!!
# Context
LOOK_BACK_CONTEXT = 1 # We add +1 because we take the last packet as the one to be compressed...
LOOK_AHEAD_CONTEXT = 1 #TIMESTEPS
CONTEXT_SIZE = LOOK_BACK_CONTEXT # Number of timesteps on contexts
CONTEXT_OUTPUT_SIZE = 30 # Taille du contexte dans la couche modèle
# Preferably < 128 as this is the size of the
# GRU layer before

# Packet
LOOK_BACK_PACKET = 8
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

# Generated dataset parameters
LEFT_PADDING = True # Padding dataset


# Name
FULL_NAME = f"LOSSLESS_PACKET{LOOK_BACK_PACKET}_{PROTO}"


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
# USEFULL CLASS/FUNCTIONS
#############################################

def standardize(x, min_x, max_x, a, b):
    """Standardize data.

    Args:
        x (np.array): data to standardize.
        min_x (int): minimum value in data.
        max_x (int): maximum value in data.
        a (int): minimum value to get in data after standardization.
        b (int): maximum value to get in data after standardization.

    Returns:
        np.array: data standardized.
    """
    x_new = (b - a) * ( (x - min_x) / (max_x - min_x) ) + a
    return x_new

def train_val_test_split(X, y, random_state, train_size=0, 
                         val_size=0, test_size=0):
    """Get indexes of holdout split.

    Args:
        X (np.array): indexes of data.
        y (np.array): indexes of target.
        random_state (int): seed used for seperation.
        train_size (int, optional): percentage of training indexes. Defaults to 0.
        val_size (int, optional): percentage of validation indexes. Defaults to 0.
        test_size (int, optional): percentage of test indexes. Defaults to 0.
        shuffle (bool, optional): shuffle the data or not. Defaults to False.
        stratify (bool, optional): stratify split. Defaults to None.

    Returns:
        tuple: indexes for training, validation and test indexes.
    """    

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
    """Apply sliding window on the data and reshape it.

    Args:
        data (np.array): the data.
        window_shape (int): size of the window applied on data.
        step (int, optional): apply the sliding. Defaults to 1.
        start_id (int, optional): first inex inside the data to 
        start the sliding windos. Defaults to None.
        end_id (int, optional): end index inside the data to stop 
        the sliding window. Defaults to None.

    Returns:
        np.array: the data sliced format to a matrix.
    """

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
    """Model for lossless compression.

    Args:
        keras (tf.keras.Model): the keras model.
    """
    def __init__(self, 
                 encoder_context,
                 ed_lossless,
                 **kwargs):
        """Constructor.

        Args:
            encoder_context (tf.keras.Model): context encoder model.
            ed_lossless (tf.keras.Model): lossless compressor model.
        """
        super(CompressorLossless, self).__init__(**kwargs)
        self.encoder_context = encoder_context
        self.ed_lossless = ed_lossless
        
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Get metrics.

        Returns:
            tf.keras.metrics: metrics.
        """
        return [
            self.loss_tracker,
        ]

    def call(self, data):
        """Compute forward pass in the deep 
        learning architecture.

        Args:
            data (np.array): data to send at input.

        Returns:
            np.array: output of the architecture.
        """
        
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
        """_summary_

        Args:
            data (np.array): data send or training.

        Returns:
            tf.keras.metrics: metrics.
        """
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
    """Build model for encoding context.

    Args:
        input_shape (np.array): input shape.
        output_size (np.array): output size.

    Returns:
        tf.keras.Model: context encoder model.
    """
    
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
    """Build embedding part of the model and the model. 
    The embedding part of the model take each data symbol
    as input.

    Args:
        input_shape (np.array): input shape.
        output_shape (np.array): output shape.
        input_dim (np.array): dimension as input.

    Returns:
        tuple: model and embedding part of the model.
    """

    output_dim = int(output_shape**0.25)
    
    inputs = keras.Input(shape=input_shape) #_symbol
    inputs_result = inputs #.values
    
    inputs_symbol = tf.slice(inputs_result, [0, 0, 0], [-1, -1, 1])
    
    inputs_symbol_shape = [k for k in tf.shape(inputs_symbol)]
    inputs_symbol = tf.reshape(
        inputs_symbol, [inputs_symbol_shape[0], inputs_symbol_shape[1]])
    
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


class DataGenerator(keras.utils.Sequence):
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
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Data didentification 
        self.packets_rank = packets_rank
        self.packets = packets
        self.headers_length = headers_length
        
        # Parameters
        self.num_sin = num_sin
        self.num_feats = self.num_sin + 1
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
        
        # Generate sinusoids
        self.sin = np.empty((self.max_length, 0), dtype=np.float64)
        for j in range(1, self.num_sin+1):
            sin_tmp = self.__gen_sin(
                F=j, Fs=self.max_length, phi=0, A=1, 
                mean=0.5, period=j, center=0.5).reshape((self.max_length, 1))
            self.sin = np.concatenate((self.sin, sin_tmp), axis=-1)    
        self.sin_seq = create_windows(
            self.sin, window_shape=self.look_back_packet, 
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
        
        # We get the data
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
            
            # Set to packet sequence
            pkt_seq[index_start:index_start+1, :, 0:1] = pkt_seq_tmp
            pkt_seq[index_start:index_start+1, :, 1:] = self.sin_seq[
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
        
        indexes = np.arange(0, pkt.shape[0])\
                    .astype(int)
        
        # Shuffle to avoid to always keep 
        # the last block not learned 
        ctx = ctx[indexes]
        pkt = pkt[indexes]
        y = y[indexes]
        
        return ctx, pkt, y


#############################################
# LAUNCH TRAINING
#############################################



# PREPARE DATA




data_raw = pd.read_csv(f"{MAIN_DIR}PROCESS/df_process_{PROTO}.csv")
arr_raw = np.load(f"./DATA/PROCESS/arr_process_{PROTO}_bit.npy", mmap_mode='r')

if (QUANTITY_PACKET is None):
    QUANTITY_PACKET = data_raw.shape[0] #max_length




# PROCESSING DATA




df_raw = data_raw\
    .reset_index(drop=True)


# CUSTOM_SIZE is present when we fix the max size
# EXTRA_SIZE is relative to the header_length
if (CUSTOM_SIZE is not None):

    max_length = int(CUSTOM_SIZE*8)
    arr = arr_raw[:, :max_length]
    print("arr.shape : ", arr.shape)

    # Remove payload
    arr_mask = np.zeros(arr.shape, dtype=np.uint8)
    for i in range(arr_mask.shape[0]):
        header_length = int(CUSTOM_SIZE*8)
        length_total = int(
            df_raw["length_total"].iloc[i] * 8)
        header_length = min(
            header_length, length_total)
        arr_mask[i, :header_length] = np.ones((1, header_length))

else:

    max_length = int(
       (df_raw["header_length"].max() + EXTRA_SIZE) * 8)
    arr = arr_raw[:, :max_length]
    print("arr.shape : ", arr.shape)

    # Remove payload
    arr_mask = np.zeros(arr.shape, dtype=np.uint8)
    for i in range(arr_mask.shape[0]):
        header_length = int(
            (df_raw["header_length"].iloc[i] + EXTRA_SIZE) * 8)
        length_total = int(
            df_raw["length_total"].iloc[i] * 8)
        header_length = min(
            header_length, length_total)
        arr_mask[i, :header_length] = np.ones((1, header_length))
    

arr = arr * arr_mask
print("arr.shape AFTER MASK : ", arr.shape)






# EXTRACT FLOWS





# Dictionary extraction with flow -> size
df_new = df_raw[['flow_id', 'timestamps']].groupby(
                ['flow_id']).min().rename(
                columns={'timestamps': 'flow_id_count'})

# Using map function for flow -> size
dict_map = df_new.to_dict()['flow_id_count']
df_raw['flow_count'] = df_raw['flow_id'].map(dict_map)

# Use map function for timestamps + number of days matching ID
df_min = df_raw[['flow_id', 'timestamps']].groupby(
    ['flow_id']).min().rename(columns={'timestamps': 'min'})
df_min['pad'] = df_min.index.values*10000000
dict_pad = df_min.to_dict()['pad']

def some_func(a, b):
    #print(dict_pad[a])
    return b+dict_pad[a]

df_raw['timestamps_update'] = df_raw[['flow_id', 'timestamps']].apply(
    lambda x: some_func(a=x['flow_id'], b=x['timestamps']), axis=1)

# We output according to timestamps (flows are autamically grouped)
df_raw = df_raw.sort_values(by=['timestamps_update'], ascending=True)
indexes_update = df_raw.index.values 

# Apply insertion to retrieve indexes cleanly
# Set index numbering
df_raw = df_raw.reset_index(drop=True)
arr_update = arr[indexes_update]

## We extract the max indexes for each flow (see index result).
indexes = df_raw.duplicated(subset=['flow_id'], keep='first')
index_val = df_raw.index[~indexes].values
index_min = index_val.copy()

## We create the min indexes for each flow (see index result) AND we do the diff
indexes = df_raw.duplicated(subset=['flow_id'], keep='last')
index_val = df_raw.index[~indexes].values
index_max = index_val.copy()

## We assign the index_min and index_max
df_new = df_raw[['flow_id']].drop_duplicates(
    subset=['flow_id'], keep='first')
df_new['index_min'] = index_min
df_new['index_max'] = index_max
df_new = df_new.set_index('flow_id')

dict_index_min = df_new[["index_min"]].to_dict()['index_min']
dict_index_max = df_new[["index_max"]].to_dict()['index_max']
df_raw['index_min'] = df_raw['flow_id'].map(dict_index_min)
df_raw['index_max'] = df_raw['flow_id'].map(dict_index_max)





# EXTRACT DATA




if (CUSTOM_SIZE is not None):
    headers_length = np.fmin(CUSTOM_SIZE,
                             df_raw['length_total'].values)
else:
    headers_length = np.fmin(df_raw['header_length'].values + EXTRA_SIZE,
                             df_raw['length_total'].values)


# We add padding
if (LEFT_PADDING):

    # Update max_length
    max_length = \
        max_length + LOOK_BACK_PACKET

    # Add padding to header length
    headers_length = headers_length + \
        (LOOK_BACK_PACKET / 8) # in bytes

    ## Define padding
    arr_update_padding = \
        np.zeros((arr_update.shape[0], 
                  LOOK_BACK_PACKET))
    arr_update = np.concatenate(
        (arr_update_padding, arr_update), 
        axis=1)


packets_rank = (df_raw.index.values - df_raw['index_min']).values
packets = arr_update
block_length = (headers_length*8 - \
                LOOK_BACK_PACKET).astype(int)

# Filter unique paquet
_, idx_unique = np.unique(packets, 
                          return_index=True,  
                          axis=0)
idx_unique = np.sort(idx_unique, axis=-1, 
            kind=None, order=None)

# Remove packet where block length
# if negative
# If size is less than the idx its
# useless to compress !!!
idx_unique_positive = np.where(
    block_length > 0)
idx_unique = np.intersect1d(
    idx_unique, idx_unique_positive)
idx_unique = np.sort(idx_unique, axis=-1, 
            kind=None, order=None)




# SELECTION EQUIPMENT 




if ("LORA" in PROTO):
    col = "device_address"
else:
    col = "ip_src"


if ((NB_EQUIP_MAX is None) or
    (NB_EQUIP_TRAIN is None)):


    idx_unique = \
        np.random.choice(idx_unique, 
                         QUANTITY_PACKET, 
                         replace=False)
    #idx_unique = idx_unique[
    #    :QUANTITY_PACKET]
    idx_unique = np.sort(idx_unique, axis=-1, 
                kind=None, order=None)

else:

    if ("LORA" in PROTO):

        list_device_address = []
        list_gateway = []

        flows_id_mac = df_raw[
            "device_address"].value_counts().index

        for f_mac in flows_id_mac:

            cond = (df_raw[
               "device_address"] == f_mac)

            device_address_tmp = df_raw[
                cond].iloc[0]['device_address']
            gateway_tmp = df_raw[
                cond].iloc[0]['gateway']

            # Check that IPs are not already present
            # among IP src
            if (device_address_tmp in list_device_address):
                pass
            else:
                list_device_address.append(
                    device_address_tmp)
                list_gateway.append(
                    gateway_tmp)

        # Select equipement
        if (MODE_EQUIP == "train"):
            list_device_address = list_device_address[:NB_EQUIP_TRAIN]
            list_gateway = list_gateway[:NB_EQUIP_TRAIN]
            NB_EQUIP_LIMIT = NB_EQUIP_TRAIN
        elif (MODE_EQUIP == "test"):
            list_device_address = list_device_address[
                NB_EQUIP_TRAIN:NB_EQUIP_MAX]
            list_gateway = list_gateway[
                NB_EQUIP_TRAIN:NB_EQUIP_MAX]
            NB_EQUIP_LIMIT = NB_EQUIP_MAX - NB_EQUIP_TRAIN
        else:
            list_device_address = list_device_address[:NB_EQUIP_MAX]
            list_gateway = list_gateway[:NB_EQUIP_MAX]
            NB_EQUIP_LIMIT = NB_EQUIP_MAX

        print("[DEBUG] list_device_address: ", list_device_address)
        print("[DEBUG] list_gateway: ", list_gateway)

        # Among the selected equipment take the flow_ids one for each
        # until the maximum number of packages is reached!
        cond = ((df_raw["device_address"].isin(list_device_address)) &
                (df_raw["gateway"].isin(list_gateway)))

        flows_id = df_raw[cond]['flow_id']\
                        .value_counts(ascending=True)\
                        .index

        # For uniformisation
        nb_device_address = len(list_device_address)

    else:

        # Find the flow_id_ip to take (five devices 
        # with different ip_src addresses)
        list_ip_src = []
        list_ip_dst = []

        flows_id_mac = df_raw[
            "flow_id_ip"].value_counts().index

        for f_mac in flows_id_mac:
            
            cond = (df_raw[
                "flow_id_ip"] == f_mac)

            ip_src_tmp = df_raw[
                cond].iloc[0]['ip_src']
            ip_dst_tmp = df_raw[
                cond].iloc[0]['ip_dst']
            
            # Check that IPs are not already present
            # among IP src
            if (ip_src_tmp in list_ip_src):
                pass
            else:
                list_ip_src.append(ip_src_tmp)
                list_ip_dst.append(ip_dst_tmp)

        # Select equipement
        if (MODE_EQUIP == "train"):
            list_ip_src = list_ip_src[:NB_EQUIP_TRAIN]
            list_ip_dst = list_ip_dst[:NB_EQUIP_TRAIN]
            NB_EQUIP_LIMIT = NB_EQUIP_TRAIN
        elif (MODE_EQUIP == "test"):
            list_ip_src = list_ip_src[
                NB_EQUIP_TRAIN:NB_EQUIP_MAX]
            list_ip_dst = list_ip_dst[
                NB_EQUIP_TRAIN:NB_EQUIP_MAX]
            NB_EQUIP_LIMIT = NB_EQUIP_MAX - NB_EQUIP_TRAIN
        else:
            list_ip_src = list_ip_src[:NB_EQUIP_MAX]
            list_ip_dst = list_ip_dst[:NB_EQUIP_MAX]
            NB_EQUIP_LIMIT = NB_EQUIP_MAX

        print("[DEBUG] list_ip_src: ", list_ip_src)
        print("[DEBUG] list_ip_dst: ", list_ip_dst)

        # Among the selected equipment take the flow_ids one for each
        # until the maximum number of packages is reached!
        cond = ((df_raw["ip_src"].isin(list_ip_src)) &
                (df_raw["ip_dst"].isin(list_ip_dst)))

        flows_id = df_raw[cond]['flow_id']\
                        .value_counts(ascending=True)\
                        .index

        # For uniformisation
        nb_device_address = len(list_ip_src)


nb_pkt = 0
i = 0
nb_equipment = 0
list_flows_id = []

# We collect a certain number of packages
# but packages of all the selected equipment 
# must be taken. Otherwise, we sample 
# with the last if
cond_while = (((nb_pkt < QUANTITY_PACKET) or 
              (nb_equipment < nb_device_address)) and 
              (i < len(flows_id)-1))
while (cond_while):
    
    cond_while = (((nb_pkt < QUANTITY_PACKET) or 
                  (nb_equipment < nb_device_address)) and 
                  (i < len(flows_id)-1))

    list_flows_id.append(flows_id[i])
    cond = (df_raw[
        'flow_id'] == flows_id[i])

    nb_pkt += df_raw[
        cond].shape[0]
    i += 1
    
    # Count number of device address
    cond = (df_raw['flow_id'].isin(list_flows_id))
    nb_equipment = df_raw[cond][
        col].value_counts().shape[0]


# Collect indexes
cond = (df_raw['flow_id'].isin(list_flows_id))
idx_unique_equipment = df_raw[cond].index
idx_unique = np.intersect1d(
    idx_unique, idx_unique_equipment)
idx_unique = np.sort(idx_unique, axis=-1, 
            kind=None, order=None)
    
# Apply correction !
# If the number of packages is greater or equal,
# then we had to hang more to be able to cover 
# all the equipment.
print("[DEBUG] len(idx_unique) : ", len(idx_unique))
print("[DEBUG] QUANTITY_PACKET : ", QUANTITY_PACKET)

if (len(idx_unique) >= QUANTITY_PACKET):
    print("[DEBUG][nb_pkt >= QUANTITY_PACKET] Apply undersampling !")
    
    idx_unique = \
        np.random.choice(idx_unique, 
                         QUANTITY_PACKET, 
                         replace=False)
    idx_unique = np.sort(idx_unique, axis=-1, 
            kind=None, order=None)


nb_equipment = df_raw.iloc[
    idx_unique][col] \
    .value_counts().shape[0]

# Check that the number of equipment items is covered
if (nb_equipment < nb_device_address):
    print("[DEBUG][WARNING !] Le nombre d'équiment choisi n'est pas présent !")
    raise Exception
    
if (nb_equipment < NB_EQUIP_LIMIT):
    print("[DEBUG][WARNING !] Le nombre d'équiment choisi est inférieur à NB_EQUIP_MAX !")
    raise Exception





# SELECT UNIQUE PACKET AND SPLIT WITH TEST




# Remove packets will be used for test
# compression
# It's useless to split test data in case 
# of test mode (equipement)
if (MODE_EQUIP != "test"):
    idx_unique_test = df_raw.iloc[idx_unique].sample(
        frac=0.1, weights='flow_id', 
        replace=False, random_state=42).index.values
    idx_unique_test = np.sort(idx_unique_test)
    print("[DEBUG] idx_unique_test: ", idx_unique_test)
    idx_unique = np.setdiff1d(
        idx_unique, idx_unique_test)
    idx_unique = np.sort(idx_unique)


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




# WE TAKE NEW LIST ID WITH UNIQUE ELEMENT (and without TEST data)




print("[DEBUG] BEFORE list_IDs.shape: ", list_IDs.shape)
print("[DEBUG] BEFORE indexes_packet.shape: ", indexes_packet.shape)
print("[DEBUG] BEFORE indexes_block.shape: ", indexes_block.shape)


# Get index for min and max element
# idx_unique is ordonned !
max_range = cumsum_block[idx_unique]
min_range = cumsum_block[idx_unique-1]

if (idx_unique[0] == 0):
    min_range[0] = 0


# New list ID
list_IDs_update = np.zeros(
    list_IDs.shape, dtype=int)
indexes_packet_update = np.zeros(
    indexes_packet.shape, dtype=int)
indexes_block_update = np.zeros(
    indexes_block.shape, dtype=int)

index_start = 0
index_end = 0

for idx_start, idx_end in zip(
    min_range, max_range):
    
    # Update index
    block_size = (idx_end - idx_start)
    index_end = index_start + block_size
    
    list_IDs_update[index_start:index_end] = \
        list_IDs[idx_start:idx_end]
    indexes_packet_update[index_start:index_end] = \
        indexes_packet[idx_start:idx_end]
    indexes_block_update[index_start:index_end] = \
        indexes_block[idx_start:idx_end]

    index_start = index_end

    
list_IDs_update = list_IDs_update[
    :index_end].astype(int)
indexes_packet_update = indexes_packet_update[
    :index_end].astype(int)
indexes_block_update = indexes_block_update[
    :index_end].astype(int)


print("[DEBUG] AFTER list_IDs_update.shape: ", list_IDs_update.shape)
print("[DEBUG] AFTER indexes_packet_update.shape: ", indexes_packet_update.shape)
print("[DEBUG] AFTER indexes_block_update.shape: ", indexes_block_update.shape)




# CREATE GENERATOR




# On shuffle pour etre sur que les adresses sont toutes couvertes
# étape de CHECK COVERAGE ne devitn plus nécessaire

idx = np.arange(0, list_IDs_update.size, dtype=int)
train_idx, val_idx, _, _ = sklearn.model_selection.train_test_split(
    idx, idx, random_state=42, 
    test_size=0.1, shuffle=SHUFFLE) # SHUFFLE


list_IDs_update_train = list_IDs_update[train_idx]
list_IDs_update_val = list_IDs_update[val_idx]

indexes_packet_update_train = indexes_packet_update[train_idx]
indexes_packet_update_val = indexes_packet_update[val_idx]

indexes_block_update_train = indexes_block_update[train_idx]
indexes_block_update_val = indexes_block_update[val_idx]



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
generator_train = DataGenerator(
    list_IDs=list_IDs_update_train, 
    indexes_packet=indexes_packet_update_train,
    indexes_block=indexes_block_update_train,
    **params) # list_IDs_update_train
'''generator_val = DataGenerator(
    list_IDs=list_IDs_update_val, 
    indexes_packet=indexes_packet_update_val,
    indexes_block=indexes_block_update_val,
    **params)'''





# DEFINE BASELINE DATAFRAME





        
# Get index get item
indexes = \
    range(list_IDs_update_train.size)
print("[DEBUG] indexes: ", indexes)

# Create DataFrame
df_baseline = pd.DataFrame(
    columns=["key", "proba"]) 

# Dictionaires proba
dict_proba = {}

# For each packet index
for k in indexes:

    # Extract item
    ctx_seq, pkt_sin_seq, \
        y_seq = generator_train.__getitem__(index=k)
    pkt_seq = pkt_sin_seq[:, :, 0]

    #print("[DEBUG] ctx_seq.shape: ", ctx_seq.shape)
    #print("[DEBUG] pkt_sin_seq.shape: ", pkt_sin_seq.shape)
    #print("[DEBUG] pkt_sin_seq.shape: ", y_seq.shape)

    # Extract key
    key = "".join(
        pkt_seq \
        .ravel() \
        .astype(int) \
        .astype(str))

    # Compute rolling proba
    y = y_seq.ravel()[0]

    # Check if the key exist
    if (key in dict_proba):
        if (y == 0):
            # Update dict_proba
            dict_proba[key] = \
                [dict_proba[key][0]+1, 
                 dict_proba[key][1]+1]
        else:
            # Update dict_proba
            dict_proba[key] = \
                [dict_proba[key][0], 
                 dict_proba[key][1]+1]
    else:
        if (y == 0):
            # Update dict_proba
            dict_proba[key] = [1, 1]
        else:
            # Update dict_proba
            dict_proba[key] = [0, 1]

# Create Dataframe to save
df_baseline['key'] = dict_proba.keys()

probas = []
for key in dict_proba.keys():
    proba = dict_proba[key][0] / \
        dict_proba[key][1]
    probas.append(proba)

# Extract huffman proba
df_baseline['proba'] = probas
            
# Save DataFrame
df_baseline.to_csv(f"{BASELINE_DIR}df_BASELINE_NAIVE_{FULL_NAME}{EXT_NAME}.csv", index=False)


