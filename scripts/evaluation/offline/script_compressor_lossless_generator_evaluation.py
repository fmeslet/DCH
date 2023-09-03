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

# Numba
from numba import jit

# Time measure
import datetime

# Multiprocessing
from multiprocessing import Pool
import multiprocessing as mp


#############################################
# SET PARAMETERS
#############################################



RESULTS_DIR = "RESULTS/"
MODELS_DIR = "MODELS/DEEP_LEARNING/"
MAIN_DIR = "./DATA/"

PROTO = "HTTP"

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
QUANTITY_PACKET = 15000 # Check if similar to training !
NUM_SIN = 8
NUM_FEATS = NUM_SIN + 1

# Filter equipment
NB_EQUIP_MAX = 10 # Set to None if no equipment
NB_EQUIP_TRAIN = 5 # Set to None if no equipment
MODE_EQUIP = "train" # or "test" or None for both
MODEL_MODE_EQUIP = "train" #"train" # or "test" or None for both
NB_EQUIP_LIMIT = NB_EQUIP_MAX # Limit to check to 
# see if we select the right number
# of equipment

# Learning parameters
SHUFFLE = False # For test generator
ALPHABET_SIZE = 2
BATCH_SIZE = 1 # For test generator

# Size added to header_length IN BYTES
EXTRA_SIZE = 0
CUSTOM_SIZE = 100 # Take the lead if extra size is define
CHECKSUM = True # True = Keep Checksum

# For evaluation parallelisation
BATCH_PACKET = 1


# Name
# min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE) if CONTEXT_OUTPUT_SIZE == 0
# else LOOK_BACK_CONTEXT == 1 or 2 or 3 or 4...etc
# easy to set LOOK_BACK_CONTEXT == 0
FULL_NAME = f"LOSSLESS_CONTEXT{min(LOOK_BACK_CONTEXT, CONTEXT_OUTPUT_SIZE)}_PACKET{LOOK_BACK_PACKET}_SIN{NUM_SIN}_{PROTO}"


# Name
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
        symlow = cumul[symbol].item() #np.asscalar(cumul[symbol])
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


def loss_fn(y_true, y_pred):
    #return 1/np.log(2) * tf.keras.metrics.categorical_crossentropy(
    #    y_true, y_pred)
    return 1/np.log(2) * tf.keras.metrics.categorical_crossentropy(
        y_true, y_pred)


def build_model_embedding_lossless(
    input_shape, output_shape, input_dim):
    output_dim = int(output_shape**0.25)
    
    inputs = keras.Input(shape=input_shape)
    
    x = inputs # ADDITION DE header shape et context
    embed = tf.keras.layers.Embedding(input_dim=input_dim, #output_shape,
                                      output_dim=output_dim,
                                      embeddings_initializer="uniform",
                                      embeddings_regularizer=None,
                                      activity_regularizer=None,
                                      embeddings_constraint=None,
                                      mask_zero=False,
                                      input_length=input_shape[-1])
    embed_output = embed(x)
    
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(64, #output_dim, #input_shape[-1],
                                activation='tanh', 
                                return_sequences=False, 
                                return_state=False))(embed_output)
    x = layers.Dense(64, 
                     activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
    x = layers.Dense(output_shape, 
                     activation="softmax")(x)
    model = keras.Model(inputs, x, name="model")
    embedding_model = keras.Model(inputs, embed_output, name="embedding_model")
    
    return model, embedding_model


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

def gen_sin(F, Fs, phi, A, mean, period, center):
    T = (1/F)*period # Period if 10/F = 10 cycles
    Ts = 1./Fs # Period sampling
    N = int(T/Ts) # Number of samples
    t = np.linspace(0, T, N)
    signal = A*np.sin(2*np.pi*F*t + phi) + center
    return signal


def next_decompression_packet(
                series, 
                context,
                header_length,
                sin,
                timesteps,
                num_feats,
                dec,
                prob,
                cumul,
                i,
                alphabet_size):

    # Si ce n'est pas la première fois que l'on arrive
    limit_size = header_length-timesteps
    if (i != limit_size):
        cumul[1:] = np.cumsum(prob*10000000 + 1)
        series[i+timesteps] = dec.read(
            cumul, alphabet_size)
        
        #if (PACKETS_GLOBAL[(i-1)+timesteps] != series[(i-1)+timesteps]):
        #    print("[DEBUG][next_decompression_packet] series EXCEPTION : \n", 
        #          series[:(i-1)+timesteps+1])
        #    print("[DEBUG][next_decompression_packet] PACKETS_GLOBAL EXCEPTION : \n", 
        #          PACKETS_GLOBAL[:(i-1)+timesteps+1])
        #    raise Exception

        i += 1
        inputs_packet = series[
            i:i+timesteps].reshape(-1, 1)
        inputs_packet = np.concatenate(
            (inputs_packet, sin[i:i+timesteps]), axis=-1)
        inputs_packet = inputs_packet.reshape((1, -1, num_feats))
        
    if (i == limit_size):   
        packet_decompress = series
        return packet_decompress
    
    return series, inputs_packet, context, header_length, sin, \
           timesteps, num_feats, dec, prob, cumul, i


def init_decompression_packet(
                packet, 
                context,
                header_length,
                max_length,
                alphabet_size,
                timesteps):
    
    num_sin = 8
    num_feats = num_sin+1
    
    # Generate sinusoide
    sin = np.empty((max_length, 0), dtype=np.float64)
    for j in range(1, num_sin+1):
        sin_tmp = gen_sin(
            F=j, Fs=max_length, phi=0, A=1, 
            mean=0.5, period=j, center=0.5).reshape((max_length, 1))
        sin = np.concatenate(
            (sin, sin_tmp), axis=-1)
    
    series = np.zeros(header_length, dtype = np.uint8)
    dec = ArithmeticDecoder(
        32, bitin=None, 
        data_compress=packet, 
        write_mode=False)
    
    prob = np.ones(alphabet_size)/alphabet_size
    cumul = np.zeros(alphabet_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
    
    i = 0
    for k in range(min(timesteps, header_length)):
        series[k] = dec.read(
            cumul, alphabet_size)
       
    # Init first input !
    inputs_packet = series[
        :timesteps].reshape(-1, 1)
    inputs_packet = np.concatenate(
        (inputs_packet, sin[:timesteps]), axis=-1)
    inputs_packet = inputs_packet.reshape(
        (1, -1, num_feats))
    #i += 1
    i = 0
      
    return series, inputs_packet, context, header_length, sin, \
          timesteps, num_feats, dec, prob, cumul, i


def next_compression_packet(packet, 
                            context,
                            packet_target,
                            header_length,
                            timesteps,
                            enc,
                            prob,
                            cumul,
                            i):
    
    
    y_original = int(np.argmax(
        packet_target[i:i+1]))

    cumul[1:] = np.cumsum(prob*10000000 + 1)
    enc.write(cumul, y_original)
    i += 1
    
    limit_size = header_length - timesteps
    if (limit_size == i):   
        enc.finish()
        return enc.data_compress 
    
    return packet, context, packet_target, \
            header_length, timesteps, enc, prob, cumul, i


def init_compression_packet(packet, 
                            context,
                            packet_target,
                            header_length,
                            alphabet_size, 
                            timesteps):    
    i = 0 # Counter for loop
    
    enc = ArithmeticEncoder(
        32, bitout=None, write_mode=False)
    prob = np.ones(alphabet_size)/alphabet_size
    cumul = np.zeros(
        alphabet_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(
        prob*10000000 + 1) 
    
    # Init with uniform law
    for j in range(timesteps):
        init_val = packet[
            0, j, 0].astype(np.uint8)
        enc.write(cumul, init_val) #X[0,j])
        
    return packet, context, packet_target, \
            header_length, timesteps, enc, prob, cumul, i


def run_parallel_compression_decompression(
            params_compression, 
            params_decompression,
            packets,
            look_back_packet,
            alphabet_size):
    
    nb_packets = len(params_compression)
    
    states_compression = np.array([False] * nb_packets)
    states_decompression = np.array([False] * nb_packets)
    
    rank_compression = np.arange(
        0, nb_packets, dtype=np.int32)
    checks_operation = [None] * nb_packets
    
    size_compress = [None] * nb_packets
    size_decompress = [None] * nb_packets
    
    
    with mp.Pool() as workers:
        
        params_compression = workers.starmap(
            init_compression_packet, 
            params_compression)
        
        # LOOP FOR COMPRESSION
        # j = compteur de boucle
        # k = compteur du rang du bit a encoder
        while (not states_decompression.all()):
            
            for j in rank_compression:
            
                if (not states_compression[j]): 
                    pkt, ctx, pkt_target, header_length, \
                      timesteps, enc, prob, cumul, k = params_compression[j]

                    limit_size = header_length - look_back_packet
                    if (k < limit_size):
                        #print("[DEBUG] ctx.shape: ", ctx.shape)
                        #print("[DEBUG] pkt[k:k+1].shape: ", pkt[k:k+1].shape)
                        prob = model.predict([ctx,
                                              pkt[k:k+1], 
                                              np.zeros((1, 2))], batch_size=1)
                        # Set alphabet_size = 2 en variable !
                        params_compression[j] = (pkt, ctx, 
                                                 pkt_target,
                                                 header_length, 
                                                 timesteps,
                                                 enc, prob, cumul, k)
                        params_compression[j] = workers.starmap(
                            next_compression_packet, (params_compression[j],))[0]
                        k += 1 # If param compression is an packet compress
                    
                    if (k == limit_size): # On init la decompression
                        states_compression[j] = True
                        size_compress[j] = len(params_compression[j])
                   
                        params_decompression[j] = (params_compression[j],
                                                   ctx,
                                                   header_length,
                                                   max_length,
                                                   alphabet_size,
                                                   look_back_packet)
                        # Map compression
                        params_decompression[j] = workers.starmap(
                            init_decompression_packet, (params_decompression[j],))[0]
             
            # Drop values inside rank compression
            # Update rank in function of evolution
            rank_compression = np.arange( # Invert mask
                0, nb_packets)[np.logical_not(states_compression)]
            rank_decompression = np.arange(
                0, nb_packets)[states_compression]        
                    
            # LOOP FOR DECOMPRESSION
            for j in rank_decompression:
                
                if ((not states_decompression[j]) and states_compression[j]): 
                    series, input_packet, ctx, header_length, sin, \
                       timesteps, num_feats, dec, prob, cumul, k = params_decompression[j]

                    limit_size = header_length - look_back_packet
                    if (k < header_length):
                        # Set alphabet_size = 2 en variable !
                        prob = model.predict([ctx,
                                              input_packet, 
                                              np.zeros((1, 2))], batch_size=1)

                        params_decompression[j] = (series, ctx,
                                                   header_length, sin,
                                                   timesteps, num_feats,
                                                   dec, prob, cumul, k,
                                                   alphabet_size)
                        params_decompression[j] = workers.starmap(
                            next_decompression_packet, 
                            (params_decompression[j],))[0]
                        k += 1 # If param compression is an packet compress

                    if (k == limit_size):
                        states_decompression[j] = True
                        size_decompress[j] = len(params_decompression[j]) # Get the packet compress now !

                        # Set packet 
                        pkt = packets[j, :header_length] #context[j:j+1, -1, :header_length]
                        check_operation =  pkt - params_decompression[j]
                        condition_check_operation = (
                            (1 in check_operation) or (-1 in check_operation))

                        # Check is decompression == original
                        if (condition_check_operation):
                            checks_operation[j] = False 
                        else:
                            checks_operation[j] = True
                            
                            
    return size_compress, size_decompress, checks_operation


def compression(model,
              packets,
              packets_rank,
              generator, 
              headers_length,
              max_length,
              range_list_IDs,
              look_back_context,
              look_ahead_context,
              look_back_packet,
              look_ahead_packet,
              num_sin,
              alphabet_size):
    
    # Define data for results
    df_results = pd.DataFrame()
    checks_operation = []
    size_compress = []
    size_decompress = []
    
    # Pre fill list for parallel computing
    params_compression = [None] * headers_length.shape[0]
    params_decompression = [None] * headers_length.shape[0]
        
    states_compression = np.array([False] * headers_length.shape[0])
    states_decompression = np.array([False] * headers_length.shape[0])
    
    rank_compression = np.arange(
        0, headers_length.shape[0], dtype=np.int32)
    checks_operation = [None] * headers_length.shape[0]
    
    size_compress = [None] * headers_length.shape[0]
    size_decompress = [None] * headers_length.shape[0]
    
    # Index batch
    idx_batch = 0
    packets_original = [None] * headers_length.shape[0]
    
    # Extraction des packets
    # et compression
    for i in range(
        range_list_IDs.shape[0]):
        
        start_idx, end_idx = \
            range_list_IDs[i, 0], range_list_IDs[i, 1]
        quantity = end_idx - start_idx # Check if good coverage !
        
        header_length = int(
            headers_length[i]*8)
        packets_original[i] = packets[i]
        
        # Init array
        idx_tmp = 0
        pkt_sin_seq = np.zeros(
            (quantity, look_back_packet, num_sin+1))
        ctx_seq = np.zeros(
            (1, look_back_context, max_length))
        pkt_target = np.zeros(
            (quantity, alphabet_size))
        
        for j in range(
            start_idx, end_idx): #idx_batch, idx_batch+quantity): # Verifier le end_idx (inclue ou non ?)

            # Get array
            ctx_seq_tmp, pkt_sin_seq_tmp, \
                pkt_target_tmp = generator \
                    .__getitem__(index=j)
            
            # Set array to packet's array
            pkt_sin_seq[idx_tmp:idx_tmp+1] = pkt_sin_seq_tmp
            pkt_target[idx_tmp:idx_tmp+1] = pkt_target_tmp
            idx_tmp += 1
            
        # On context is sufficient useless 
        # to repear !
        ctx_seq[0:1] = ctx_seq_tmp
        idx_batch = idx_batch + quantity

        # Array for params compression
        params_compression[i] = (pkt_sin_seq,
                                 ctx_seq,
                                 pkt_target,
                                 header_length,
                                 alphabet_size,
                                 look_back_packet)
        
    size_compress, size_decompress, checks_operation = \
        run_parallel_compression_decompression(
                params_compression=params_compression, 
                params_decompression=params_decompression,
                packets=packets,
                look_back_packet=look_back_packet,
                alphabet_size=alphabet_size)
        
    df_results['size_decompress'] = size_decompress
    df_results['size_compress'] = size_compress
    df_results['check_operation'] = checks_operation
    df_results['headers_length'] = headers_length*8 # Transform to bit
    df_results['packets_rank'] = packets_rank
    
    return df_results


# GENERATOR UPDATE pour prendre la notion d'index
# sans sequentialité !
class DataGenerator(keras.utils.Sequence):
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
                 
                 batch_size=32,
                 num_sin=8,
                 alphabet_size=2, 
                 shuffle=True):
        'Initialization'
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
        self.num_feats = self.num_sin + 1
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
        
        # Generate sinusoide
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
    
    
    def __get_indexes(
        self, indexes):
        
        indexes_packet = np.zeros(
            (indexes.size,), dtype=int)
        indexes_block = np.zeros(
            (indexes.size,), dtype=int)
        
        for i, idx in enumerate(indexes):
            
            # Found begin and end
            rank_min = np.where(
                idx+1 > self.indexes_min_packet)[0]
            rank_max = np.where(
                idx <= self.indexes_max_packet)[0]
            
            # Get index of packet associated with block
            indexes_packet[i] = np.intersect1d(
                rank_min, rank_max).astype(int)[0]
            
            # Get index block
            #print("[DEBUG] indexes_packet: ", indexes_packet)
            indexes_block[i] = (idx - \
                self.indexes_min_packet[
                    indexes_packet[-1]]).max()
        
        return indexes_packet, indexes_block
    

    def __getitem__(
        self, index):
        'Generate one batch of data'
        # On met à jour l'index avec le coté random !
        
        # Get index of block
        index_start = self.indexes[
            index] * self.batch_size
        index_end = index_start + self.batch_size
        indexes_random = self.list_IDs[
            index_start:index_end] # Correspond à l'index block !
        
        # Get indexes packet and block
        indexes_packet, indexes_block = self.__get_indexes(
            indexes=indexes_random)
        
        # On prend le min et on ajoute !
        # On récupère les données
        ctx, pkt, y = self.__data_generation(
                indexes_packet, indexes_block)

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
            (nb_repeat, self.look_back_packet, self.num_feats))
        
        y_seq = np.zeros(
            (nb_repeat, self.alphabet_size))
        
        index_start = int(0)
        for idx_pkt, idx_block in zip(
            indexes_packet, indexes_block):
            
            #nb_block = int((self.headers_length[i]*8) - \
            #     self.look_back_packet)
            #nb_block = 1
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
        np.random.shuffle(indexes)
        
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



# REmove packet with wrong CRC
condition = (data_raw['crc_status'] == 1)
df_raw = data_raw[condition] \
    .reset_index(drop=True).copy()

# Extraction d'un dictonnaire avec flux -> taille
df_new = df_raw[['flow_id', 'timestamps']].groupby(
                ['flow_id']).min().rename(columns={'timestamps': 'flow_id_count'})

# Utilisation de map function pour flux -> taille
dict_map = df_new.to_dict()['flow_id_count']
df_raw['flow_count'] = df_raw['flow_id'].map(dict_map)

# Utilisation de map function pour timestamps + nombre de jour qui correpsond à l'ID
df_min = df_raw[['flow_id', 'timestamps']].groupby(
    ['flow_id']).min().rename(columns={'timestamps': 'min'})
df_min['pad'] = df_min.index.values*100000
    #'min'].map(lambda x : np.random.randint(0, 5000, size=(1,))[0])
dict_pad = df_min.to_dict()['pad']

def some_func(a, b):
    #print(dict_pad[a])
    return b+dict_pad[a]

df_raw['timestamps_update'] = df_raw[['flow_id', 'timestamps']].apply(
    lambda x: some_func(a=x['flow_id'], b=x['timestamps']), axis=1)

# On sort en fonction de la timestamps (les flux sont autamiquement groupe)
df_raw = df_raw.sort_values(by=['timestamps_update'], ascending=True)
indexes_update = df_raw.index.values 

# J'applique l'insertion pour récuperer les index proprement
# Mettre les la numerotation des index
df_raw = df_raw.reset_index(drop=True)
arr_update = arr[indexes_update]

## On extrait les index max de chaque flux (voir index result)
indexes = df_raw.duplicated(subset=['flow_id'], keep='first')
index_val = df_raw.index[~indexes].values
index_min = index_val.copy()

## On créer les index min de chaque flux (voir index result) ET on fait la diff
indexes = df_raw.duplicated(subset=['flow_id'], keep='last')
index_val = df_raw.index[~indexes].values
index_max = index_val.copy()

## On attribue les index_min et les index_max
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

            # Verifier que les IP ne sont pas deja présentes
            # parmis l'IP src
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

        # Parmis les equipement selectionner prendre les flow_ids un pour chaque
        # jusqu'a que le nombre max de paquet soit atteint !
        cond = ((df_raw["device_address"].isin(list_device_address)) &
                (df_raw["gateway"].isin(list_gateway)))

        flows_id = df_raw[cond]['flow_id']\
                        .value_counts(ascending=True)\
                        .index

        # For uniformisation
        nb_device_address = len(list_device_address)

    else:

        # Trouver les flow_id_mac a prendre (cinq equipement avec des adress ip_src différente)
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
            
            # Verifier que les IP ne sont pas deja présentes
            # parmis l'IP src
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

        # Parmis les equipement selectionner prendre les flow_ids un pour chaque
        # jusqu'a que le nombre max de paquet soit atteint !
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

# On collecte un certains nombre de paquet
# mais il faut que des paquets de tous les equipements 
# selectionné soit pris. Sinon on echantillonne
# avec le dernier if
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
# Si le nb de paquet est supérieur ou égale
# alors on a du en pendre plus pour pouvoir
# couvrir tous les equipements
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

# Verifier que le nombre de equipements sont couvert
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





# WE TAKE NEW LIST ID WITH UNIQUE ELEMENT !




print("[DEBUG] BEFORE list_IDs.shape: ", list_IDs.shape)
print("[DEBUG] BEFORE indexes_packet.shape: ", indexes_packet.shape)
print("[DEBUG] BEFORE indexes_block.shape: ", indexes_block.shape)



# /!\ TO REMOVE !


#idx_unique_test = idx_unique_test[:5]

# Get index for min and max element
# idx_unique is ordonned !
max_range = cumsum_block[idx_unique_test]
min_range = cumsum_block[idx_unique_test-1]

if (idx_unique_test[0] == 0):
    min_range[0] = 0

# New list ID
list_IDs_test = np.zeros(
    list_IDs.shape)
indexes_packet_test = np.zeros(
    indexes_packet.shape, dtype=int)
indexes_block_test = np.zeros(
    indexes_block.shape, dtype=int)


index_start = 0
index_end = 0

range_list_IDs_test = np.zeros(
    (list_IDs.shape[0], 2))
range_index_start = 0
range_index_end = 0


for idx_start, idx_end in zip(
    min_range, max_range):
    
    # Update index
    range_index_end = range_index_start + 1    
    range_list_IDs_test[range_index_start:range_index_end] = [
        list_IDs[idx_start], list_IDs[idx_end-1]+1] # [0, +inf[
    range_index_start = range_index_end
    
    # Update index
    block_size = (idx_end - idx_start)
    index_end = index_start + block_size
    
    list_IDs_test[index_start:index_end] = \
        list_IDs[idx_start:idx_end]
    indexes_packet_test[index_start:index_end] = \
        indexes_packet[idx_start:idx_end]
    indexes_block_test[index_start:index_end] = \
        indexes_block[idx_start:idx_end]

    index_start = index_end

    
list_IDs_test = list_IDs_test[
    :index_end].astype(int)
indexes_packet_test = indexes_packet_test[
    :index_end].astype(int)
indexes_block_test = indexes_block_test[
    :index_end].astype(int)

range_list_IDs_test = range_list_IDs_test[
    :range_index_end].astype(int)


print("[DEBUG] AFTER list_IDs_test.shape: ", list_IDs_test.shape)
print("[DEBUG] AFTER indexes_packet_test.shape: ", indexes_packet_test.shape)
print("[DEBUG] AFTER indexes_block_test.shape: ", indexes_block_test.shape)




# SET TEST GENERATOR 




# Set parameters
params = {'look_back_context': LOOK_BACK_CONTEXT,
          'look_ahead_context': LOOK_AHEAD_CONTEXT, # Par default on ne sais qu'avec 1
          'look_back_packet': LOOK_BACK_PACKET,
          'look_ahead_packet': LOOK_AHEAD_PACKET, # Par default on ne sais qu'avec 1
         
          'packets_rank': packets_rank,
          'packets': packets,
          'headers_length': headers_length,
          'max_length': max_length,
         
          'batch_size': BATCH_SIZE,
          'num_sin': NUM_SIN,
          'alphabet_size': ALPHABET_SIZE, 
          'shuffle': SHUFFLE}


# Generators
# Index = index de bloc de batch !
# Example:
# Si len(listID) = 264
# alors index MAX = 132 si BATCH_SIZE = 2
generator_test = DataGenerator(
    list_IDs=list_IDs_test, 
    indexes_packet=indexes_packet_test,
    indexes_block=indexes_block_test,
    **params)




# LOAD MODEL




# Name
if (CHECKSUM):
    if (CUSTOM_SIZE is not None):
        EXT_NAME_MODEL = f"_WITH_CHECKSUM_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_EQUIP{MODEL_MODE_EQUIP}"
    else:
        EXT_NAME_MODEL = f"_WITH_CHECKSUM_EXTRA_SIZE{EXTRA_SIZE}_MODE_EQUIP{MODEL_MODE_EQUIP}"

else:
    if (CUSTOM_SIZE is not None):
        EXT_NAME_MODEL = f"_CUSTOM_SIZE{CUSTOM_SIZE}_MODE_EQUIP{MODEL_MODE_EQUIP}"
    else:
	EXT_NAME_MODEL = f"_EXTRA_SIZE{EXTRA_SIZE}_MODE_EQUIP{MODEL_MODE_EQUIP}"


encoder_context = tf.keras.models.load_model(f"{MODELS_DIR}ENCODER_CONTEXT_{FULL_NAME}{EXT_NAME_MODEL}.h5", 
                                              custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
ed_lossless = tf.keras.models.load_model(f"{MODELS_DIR}ED_{FULL_NAME}{EXT_NAME_MODEL}.h5", 
                                          custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
embedder = tf.keras.models.load_model(f"{MODELS_DIR}EMBEDDER_{FULL_NAME}{EXT_NAME_MODEL}.h5", 
                                      custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
model = CompressorLossless(encoder_context=encoder_context, 
                           ed_lossless=ed_lossless)



# START COMPRESSION


gc.collect()
df_results = compression(model=model,
                          generator=generator_test, 
                          packets=packets[idx_unique_test], 
                          packets_rank=packets_rank[idx_unique_test],
                          headers_length=headers_length[idx_unique_test], 
                          max_length=max_length,
                          range_list_IDs=range_list_IDs_test,
                          look_back_context=LOOK_BACK_CONTEXT,
                          look_ahead_context=LOOK_AHEAD_CONTEXT,
                          look_back_packet=LOOK_BACK_PACKET,
                          look_ahead_packet=LOOK_AHEAD_PACKET,
                          num_sin=NUM_SIN,
                          alphabet_size=ALPHABET_SIZE)

df_results['idx_unique'] = idx_unique_test

###############
# SAVE RESULT
###############

df_results.to_csv(f"{RESULTS_DIR}df_{FULL_NAME}{EXT_NAME}.csv", index=False)

end_time = datetime.datetime.now()

print("TIME DIFF : ", end_time-start_time)
