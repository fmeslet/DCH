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


HUFFMAN_DIR = "MODELS/HUFFMAN/DEEP_LEARNING/"
HUFFMAN_PARTS_DIR = HUFFMAN_DIR + "PARTS/"
FIELDS_DIR = "RESULTS/FIELDS/DEEP_LEARNING/"
MODELS_DIR = "MODELS/DEEP_LEARNING/"
MAIN_DIR = "./DATA/"

PROTO = "SMTP"

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
CUT_VALUE = 2 #For array of index possition
KEEP_ERROR = True # Change EXT_NAME, remove KEEP ERROR !
OPTIMAL = True
SELECTIVE = True # if True he most rank bit is used (only one)
MAX_RANK = 29 # Max rank level explore with num_rank (not starting from 0)

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



# PREPARE DATA




data_raw = pd.read_csv(f"{MAIN_DIR}PROCESS/df_process_{PROTO}.csv")
arr_raw = np.load(f"./DATA/PROCESS/arr_process_{PROTO}_bit.npy", mmap_mode='r')

# Import df fields
if (LEFT_PADDING):
    df_fields = pd.read_csv(f"{FIELDS_DIR}df_FIELDS_{FULL_NAME}{EXT_NAME}_LEFT_PADDING.csv")
else:
    df_fields = pd.read_csv(f"{FIELDS_DIR}df_FIELDS_{FULL_NAME}{EXT_NAME}.csv")

#df_fields = df_fields\
#        .set_index("list_IDs")


if (QUANTITY_PACKET is None):
    QUANTITY_PACKET = data_raw.shape[0] #max_length




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





# Extraction d'un dictonnaire avec flux -> taille
df_new = df_raw[['flow_id', 'timestamps']].groupby(
                ['flow_id']).min().rename(
                columns={'timestamps': 'flow_id_count'})

# Utilisation de map function pour flux -> taille
dict_map = df_new.to_dict()['flow_id_count']
df_raw['flow_count'] = df_raw['flow_id'].map(dict_map)

# Utilisation de map function pour timestamps + nombre de jour qui correpsond à l'ID
df_min = df_raw[['flow_id', 'timestamps']].groupby(
    ['flow_id']).min().rename(columns={'timestamps': 'min'})
df_min['pad'] = df_min.index.values*10000000
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
df_raw = df_raw.reset_index(drop=False)
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

        # Trouver les flow_id_ip a prendre (cinq equipement avec des adress ip_src différente)
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





# Extract part of array
df_raw_tmp = df_raw[['index', 'index_min']]


# Extract relative position of packet
df_raw_tmp['index_rltv_flow'] = df_raw_tmp.index - \
                df_raw_tmp['index_min']


# Extract dict correspondance index <-> idx_rltv_flow
dict_index_index_rltv_flow = df_raw_tmp[
                ['index', 'index_rltv_flow']] \
        .set_index('index')['index_rltv_flow'] \
        .to_dict()

# Set relative position to flow
df_fields['index_rltv_flow'] = df_fields['index_packet'] \
            .map(dict_index_index_rltv_flow)


# Set index relative to packet
df_fields['index_rltv_packet'] = df_fields['index_block']


# Set max fields
max_fields = (LOOK_BACK_CONTEXT * max_length) + \
                LOOK_BACK_PACKET


# Remove wrong prediction
if (not KEEP_ERROR):
    cond = (df_fields['pred_true'] == df_fields['pred_pred'])
    df_fields = df_fields[cond]





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

list_IDs_update_train_undersample = \
            df_fields_tmp['list_IDs'].values.astype(int)
indexes_packet_update_train_undersample = \
            df_fields_tmp['index_packet'].values.astype(int)
indexes_block_update_train_undersample = \
            df_fields_tmp['index_block'].values.astype(int)

print("[DEBUG] list_IDs_update_train_undersample.shape : ", list_IDs_update_train_undersample.shape)
print("[DEBUG] indexes_packet_update_train_undersample.shape : ", indexes_packet_update_train_undersample.shape)
print("[DEBUG] indexes_block_update_train_undersample.shape : ", indexes_block_update_train_undersample.shape)

print("[DEBUG] list_IDs_update_train_undersample : ", list_IDs_update_train_undersample)
print("[DEBUG] indexes_packet_update_train_undersample : ", indexes_packet_update_train_undersample)
print("[DEBUG] indexes_block_update_train_undersample : ", indexes_block_update_train_undersample)

# Generators
generator = DataGeneratorContinuous(
          list_IDs=list_IDs_update_train_undersample, 
          indexes_packet=indexes_packet_update_train_undersample,
          indexes_block=indexes_block_update_train_undersample,
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
 

        '''if (SELECTIVE):

            if (OPTIMAL):
                if (KEEP_ERROR):
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_OPTIMAL_{j}_{i}.csv", index=False)
                else:
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_SELECTIVE_OPTIMAL_{j}_{i}.csv", index=False)
            else:
                if (KEEP_ERROR):
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_{j}_{i}.csv", index=False)
                else:
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_SELECTIVE_{j}_{i}.csv", index=False)

        else:
            
            if (OPTIMAL):
                if (KEEP_ERROR):
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_OPTIMAL_{j}_{i}.csv", index=False)
                else:
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_OPTIMAL_{j}_{i}.csv", index=False)
            else:
                if (KEEP_ERROR):
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_{j}_{i}.csv", index=False)
                else:
                    df_huffman.to_csv(f"{HUFFMAN_PARTS_DIR}df_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_{j}_{i}.csv", index=False)'''
            


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

# Set extension
arr_name += f".npy"


# Save array
print("[DEBUG] arr_name: ", arr_name)
print("[DEBUG] array_index_pos: ", array_index_pos)
np.save(arr=array_index_pos, file=arr_name)

# np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_OPTIMAL.npy")



'''if (SELECTIVE):

    if (OPTIMAL):
        if (KEEP_ERROR):
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_OPTIMAL.npy")
        else:
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_SELECTIVE_OPTIMAL.npy")
    else:
        if (KEEP_ERROR):
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_SELECTIVE.npy")
        else:
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_SELECTIVE.npy")

else:

    if (OPTIMAL):
        if (KEEP_ERROR):
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}_OPTIMAL.npy")
        else:
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}_OPTIMAL.npy")
    else:
        if (KEEP_ERROR):
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_KEEP_ERROR_{CUT_VALUE}.npy")
        else:
            np.save(arr=array_index_pos, file=f"{HUFFMAN_DIR}arr_index_pos_HUFFMAN_{FULL_NAME}{EXT_NAME}_{CUT_VALUE}.npy")'''
