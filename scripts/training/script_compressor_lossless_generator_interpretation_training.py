#!/usr/bin/python3
#-*-coding: utf-8-*-

#############################################
# IMPORTATIONS
#############################################

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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# Personnal functions
# import functions

#############################################
# SET PARAMETERS
#############################################

RESULTS_DIR = "RESULTS/INTERPRETATION/"
MODELS_DIR = "MODELS/INTERPRETATION/"
MAIN_DIR = "./DATA/"

# Set keras types
tf.keras.backend.set_floatx('float64')

# Parameters
LOOK_BACK_CONTEXT = 1
LOOK_AHEAD_CONTEXT = 1 #TIMESTEPS
CONTEXT_SIZE = LOOK_BACK_CONTEXT # Nombre timesteps sur les contexts
CONTEXT_OUTPUT_SIZE = 30 # Size of contexte in model layer
# De preference < 128 car c'est la taille de la
# couche GRU avant

LOOK_BACK_PACKET = 16 # Need to be multiple of 8 for padding
LOOK_AHEAD_PACKET = 1
NUM_SIN = 8
NUM_FEATS = NUM_SIN + 1

# Learning parameters
EPOCHS = 7
SHUFFLE = False
ALPHABET_SIZE = 2
BATCH_SIZE = 1 #512 #2048

# Size added to header_length IN BYTES
EXTRA_SIZE = 0
CUSTOM_SIZE = 100 # Take the lead if extra size is define
CHECKSUM = True # True = Keep Checksum

# Generated dataset parameters
MAX_PACKET_RANK = 3 # Max rank support for packet
NB_BITS = 16*2 # Multiple for field inversion
NB_ROWS = 4000*MAX_PACKET_RANK # Multiple for the flow id NB_ROWS = x * MAX_PACKET_RANK
MODE_DATASET = "inversion" # "random", "checksum", "checksum34", "fixed", "inversion", "counter", "fixed_flow", "combinaison"
LEFT_PADDING = False # Padding dataset


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
                         val_size=0, test_size=0, shuffle=False,
                         stratify=None):
    
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
                                shuffle=shuffle, stratify=stratify)

    # Get data test from val
    X = X[val_idx]
    y = y[val_idx]
    val_idx, test_idx, _, _ = sklearn.model_selection.train_test_split(X, y,
                                random_state=random_state, test_size=0.5, 
                                shuffle=shuffle, stratify=stratify[val_idx])
    
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
    
    x = encoder_inputs 
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


# PREPARE DATASETS



# For checksum AND combinaison !
# From : https://www.thegeekstuff.com/2012/05/ip-header-checksum/

def sum_checksum(
    array_bit_a, array_bit_b):
    """Sum two arrays of bits.

    Args:
        array_bit_a (np.array): first array.
        array_bit_b (np.array): second array.

    Returns:
        np.array: _description_
    """
    
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
    """Convert an integer into a binary value.

    Args:
        x (np.array): _description_
        size (int, optional): size of the integer in. Defaults to 8.

    Returns:
        _type_: _description_
    """
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




# CREATE GENERATOR




# Shuffle to make sure all addresses are covered
# CHECK COVERAGE step no longer necessary

idx = np.arange(0, list_IDs.size, dtype=int)
train_idx, val_idx, _, _ = sklearn.model_selection.train_test_split(
    idx, idx, random_state=42, 
    test_size=0.1, shuffle=SHUFFLE) # SHUFFLE


list_IDs_train = list_IDs[train_idx]
list_IDs_val = list_IDs[val_idx]

indexes_packet_train = indexes_packet[train_idx]
indexes_packet_val = indexes_packet[val_idx]

indexes_block_train = indexes_block[train_idx]
indexes_block_val = indexes_block[val_idx]



params = {'look_back_context': LOOK_BACK_CONTEXT,
          'look_ahead_context': LOOK_AHEAD_CONTEXT,
          'look_back_packet': LOOK_BACK_PACKET,
          'look_ahead_packet': LOOK_AHEAD_PACKET, 
         
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
    list_IDs=list_IDs_train, 
    indexes_packet=indexes_packet_train,
    indexes_block=indexes_block_train,
    **params) # list_IDs_train
generator_val = DataGenerator(
    list_IDs=list_IDs_val, 
    indexes_packet=indexes_packet_val,
    indexes_block=indexes_block_val,
    **params)






# CREATE MODEL AND TRAIN





gc.collect()
cbs = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                         factor=0.5,
                         patience=1,
                         min_lr=1e-7,
                         min_delta=0.1,
                         verbose=0,
                         skip_mismatch=True)]

if (LEFT_PADDING):
    encoder_context = build_model_context_lossless(
        input_shape=(CONTEXT_SIZE, max_length-LOOK_BACK_PACKET), 
        output_size=CONTEXT_OUTPUT_SIZE)
else:
    encoder_context = build_model_context_lossless(
        input_shape=(CONTEXT_SIZE, max_length), 
        output_size=CONTEXT_OUTPUT_SIZE)

ed_lossless, embedder = build_model_lossless(
                    input_shape=(LOOK_BACK_PACKET, CONTEXT_OUTPUT_SIZE+NUM_FEATS), 
                    output_shape=ALPHABET_SIZE,
                    input_dim=ALPHABET_SIZE)

model = CompressorLossless(encoder_context=encoder_context,
                           ed_lossless=ed_lossless)

model.compile(optimizer=keras.optimizers.Adam(1e-4), 
              run_eagerly=False)

history = model.fit_generator(generator=generator_train,
                              validation_data=generator_val,
                              epochs=EPOCHS,
                              use_multiprocessing=True,
                              callbacks=cbs)

print(history.history)


#############################################
# SAVE MODEL and VALUES
#############################################


encoder_context.save(f"{MODELS_DIR}ENCODER_CONTEXT_{FULL_NAME}{EXT_NAME}.h5")
ed_lossless.save(f"{MODELS_DIR}ED_{FULL_NAME}{EXT_NAME}.h5")
embedder.save(f"{MODELS_DIR}EMBEDDER_{FULL_NAME}{EXT_NAME}.h5")
