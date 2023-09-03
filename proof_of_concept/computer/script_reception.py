#!/home/fmeslet/anaconda3/bin/python3.8
#-*-coding: utf-8 -*-


###############
# IMPORTATIONS
###############

# Data manipulation
import numpy as np
import pandas as pd

# Scapy
from scapy.all import *
from scapy.compat import bytes_encode

# Sytem manipulation
import os

# Frequency table
from frequency_table import *

# Context mapper
from context_mapper import *

# Decompressor
from decompressor import *

# Arithmetic encoder
# from arithmetic_ed import *


#############################################
# SET PARAMETERS
#############################################

# Context
LOOK_BACK_CONTEXT = 2 # On rajoute +1 car on prend le dernier paquet comme paquet à compresser...
LOOK_AHEAD_CONTEXT = 1 #LOOK_BACK_PACKET
INDEX_POINTER = 0
QUANTITY_CONTEXT = 10000
CONTEXT_SIZE = 2 # Nombre LOOK_BACK_PACKET sur les contexts
CONTEXT_OUTPUT_SIZE = 30 # Size of contexte in model layer

# Packet
LOOK_BACK_PACKET = 16
LOOK_AHEAD_PACKET = 1
NUM_SIN = 8
NUM_FEATS = NUM_SIN+1
ALPHABET_SIZE = 2
MAX_LENGTH = 800 #464 # Help to define max context length


PROTO = "SMTP"
CUT_VALUE = 6

FULL_NAME = f"LOSSLESS_CONTEXT{LOOK_BACK_CONTEXT}_PACKET{LOOK_BACK_PACKET}_SIN{NUM_SIN}_{PROTO}_WITH_CHECKSUM_CUSTOM_SIZE100_MODE_EQUIPtrain_KEEP_ERROR_{CUT_VALUE}_SELECTIVE_OPTIMAL_LEFT_PADDING"
#EXT_NAME = f"_HUFFMAN_TABLE_PARALLEL_COMPRESSION_{CUT_VALUE}_FROM_END" #_IP_ADDRESS_2"

# (wlan.addr == 50:02:91:9c:88:8c)
MAC_ADDR_FILTER = "50:02:91:9c:88:8c"
IP_VERSION = 4


################
# FUNCTIONS
################


def packet_filter (pkt) :
        if ((pkt.addr2 == MAC_ADDR_FILTER) and 
            ('Raw' in pkt)):
            return True
        else:
            return False


def gen_sin(F, Fs, phi, A, mean, period, center):
    T = (1/F)*period # Period if 10/F = 10 cycles
    Ts = 1./Fs # Period sampling
    N = int(T/Ts) # Number of samples
    t = np.linspace(0, T, N)
    signal = A*np.sin(2*np.pi*F*t + phi) + center
    return signal


def transform_packet_bytes_int(
        packet_bytes, length=1522):
    packet_int = [int(byte) for byte in packet_bytes]
    packet_int_array = np.array(packet_int)

    if (length is not None):
        packet_int_array_pad = np.lib.pad(packet_int_array,
                                (0,length-packet_int_array.shape[0]),
                                'constant', constant_values=(0))

        packet_int_array_pad = np.reshape(packet_int_array_pad, (length, 1))
        return packet_int_array_pad
    else:
        return packet_int_array


def transform_packet_bytes_bit(
        packet_bytes, length=1522):
    payload_bits = ''.join(format(byte, '08b') for byte in packet_bytes)

    def split(bits):
        return [bit for bit in bits]

    packet_bit_array = np.array(split(payload_bits), dtype=np.int8)
    
    if (length is not None):
        packet_bit_array_pad = np.lib.pad(packet_bit_array, 
                                (0,length-packet_bit_array.shape[0]), 
                                'constant', constant_values=(0))

        packet_bit_array_pad = np.reshape(packet_bit_array_pad, (length, 1))
        return packet_bit_array_pad.astype(np.int8)
    else:
        return packet_bit_array


def convert_bit_int(array):
    length = array.shape[-1]
    value = 0

    for i in range(length):
        value += (2**(
            length-i-1)) * array[i]
    
    return value


def transform_packet_bit_int(array, 
                             size=8):
    length = array.shape[-1]
    array_int = []
    value = 0

    for i in range(0, length, size):
        for j in range(0, size):
            value += (2**(size-1-j)) \
                    * array[i+j]

        array_int.append(value)
        value = 0
    
    return np.array(array_int)



################
# SET PARAMETERS
################


if __name__ == '__main__':

    # LOAD ARRAY

    df_huffman_groupby = pd.read_csv(f"df_HUFFMAN_{FULL_NAME}.csv")
    print("[DEBUG] df_huffman_groupby.columns: ", df_huffman_groupby.columns)
    #df_huffman_groupby = df_huffman_groupby.drop(["Unnamed: 0"], axis=1)
    array_index_pos = np.load(f"arr_index_pos_HUFFMAN_{FULL_NAME}.npy")


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
    
    #print("[DEBUG] df_huffman_groupby : ", df_huffman_groupby)

    # LOAD model

    #df_context = pd.DataFrame(
    #    columns=['ip_src', 'ip_dst', 'port_src', 'port_dst'])
    #arr_context = np.zeros((0, 2, 464), dtype=np.uint32) # Set la max length

    frequency_table = FrequencyTable(df_table=df_huffman_groupby,
                                     index_pos=array_index_pos.astype(np.int32),
                                     alphabet_size=ALPHABET_SIZE,
                                     cut_value=CUT_VALUE)

    # LOOP for capture

    context_mapper = ContextMapper(size_context=CONTEXT_SIZE,
                                   size_element=MAX_LENGTH)

    decompressor = Decompressor(context_mapper=context_mapper,
                                frequency_table=frequency_table,
                                alphabet_size=ALPHABET_SIZE,
                                overhead=LOOK_BACK_PACKET)


    '''def get_layers(packet):
        layer = []
        for i in packet.layers():
          name = str(i).split('.')[-1][:-2]
          layer.append(name)
        return layer'''
    
    test = True
    counter = 0
    while (test):

        print("")

        ## Capture packet

        packet = sniff(iface="mon1", 
		               count=1,
                       lfilter=packet_filter)
        packet = packet[0]
        #print("[DEBUG] packet.nsummary() : ") 
        #packet.nsummary()
        #print("[DEBUG] packet : ", packet.show())
        packet_bytes = bytes_encode(packet['Raw'])
        #print("[DEBUG] packet_bytes: ", packet_bytes)

        packet_ether_bytes = bytes_encode(packet["Dot11"])

        mac_src = transform_packet_bytes_int(
             packet_ether_bytes[9:16], length=None)
        mac_dst = transform_packet_bytes_int(
             packet_ether_bytes[15:22], length=None)
        print("[DEBUG] mac_src : ", mac_src) 
        print("[DEBUG] mac_dst : ", mac_dst)
        key_array =  mac_src.tolist() + \
                mac_dst.tolist()
        print("[DEBUG] key_array : ", key_array)

        ## Convert from hexa to int
        packet_bit = transform_packet_bytes_bit(
                    packet_bytes, length=None)
        packet_bit = packet_bit[7*8:] # Remove LLC part no detect
        print("[DEBUG] packet_bit : ", packet_bit, 
              " // packet_bit.shape : ", packet_bit.shape)

        # For printing ! TO REMOVE
        packet_int = transform_packet_bytes_int(
                    packet_bytes, length=None)
        packet_int = packet_int[7:] # Remove LLC part no detect
        print("[DEBUG] packet_int : ", packet_int, 
              " // packet_int.shape : ", packet_int.shape)

        # On extrait le context niveau MAC
        #print("[DEBUG] packet['LLC'] : ", packet['LLC'].show())
        llc_layer_bytes = bytes_encode(packet['LLC'].load)
        llc_layer_int = transform_packet_bytes_int(
                    llc_layer_bytes, length=None)
        llc_layer_type = llc_layer_int[5:7]

        # Add padding
        padding_bit = np.zeros((LOOK_BACK_PACKET,), 
                        dtype=np.uint8) 
        packet_bit = np.concatenate(
                (padding_bit, packet_bit))

        print("[DEBUG] FIT IN DECOMPRESSOR packet_bit: ", packet_bit)
        print("[DEBUG] FIT IN DECOMPRESSOR packet_bit.shape: ", packet_bit.shape)
        
        # Fit the packet
        decompressor.fit(
            data_compress=packet_bit)

        print("[DEBUG] FIT IN DECOMPRESSOR packet_bit: ", packet_bit)

        #print("[DEBUG] dec.data_decompress INIT : ", dec.data_decompress)


        # Artificially set the context

        #print("[DEBUG] size_context shape : ", size_context.shape)
        #print("[DEBUG] data_context shape : ", data_context.shape)
         
        print("[DEBUG] llc_layer_type : ", llc_layer_type)

        # Apply treatment if Ipv6 of IPv4

        if (llc_layer_type.sum() == 8): # Type = IPv4 [8, 0]

            print("[DEBUG][llc_layer_type.sum() == 8] Je suis dedans")

            ## Start decompression until length field
            print("[DEBUG] decompressor.data_decompress : ", 
                decompressor.data_decompress)

            decompressor.decompress(key_array=key_array,
                                    start_range=0, 
                                    end_range=4*8 #-LOOK_BACK_PACKET 
                                    # -LOOK_BACK_PACKET directement intégré dans la fonction
                                    )
            start_range = 4*8 # - LOOK_BACK_PACKET
            #print("[DEBUG] decompressor.data_decompress shape : ", 
            #        len(decompressor.data_decompress))
            #print("[DEBUG] decompressor.data_decompress : ", 
            #        decompressor.data_decompress)
            
            dec_decompress_int = transform_packet_bit_int(
                                            np.array(decompressor.data_decompress), 
                                            size=8)
            #print("[DEBUG] dec_decompress_int : ", dec_decompress_int)

        elif (llc_layer_type.sum() == 355): # Type = IPv6 [134, 221]

            print("[DEBUG][llc_layer_type.sum() == 355] Je suis dedans")
            
            ## Start decompression until length field
            decompressor.decompress(key_array=key_array,
                                    start_range=0, 
                                    end_range=5*8#-LOOK_BACK_PACKET
                                    )
            start_range = 5*8 # - LOOK_BACK_PACKET

        else:
            print("[WARNING] LLC Type field value unknown !")

        dec_decompress_int = transform_packet_bit_int(
                np.array(decompressor.data_decompress), 
                size=8)
        print("[DEBUG] decompressor.data_decompress : ", decompressor.data_decompress)
        print("[DEBUG] dec_decompress_int : ", dec_decompress_int)


        ## Extract length

        ip_layer_length_bit = np.array(decompressor.data_decompress[-2*8:])
        ip_layer_length = convert_bit_int(
                ip_layer_length_bit)

        print("[DEBUG] ip_layer_length_bit : ", ip_layer_length_bit)
        print("[DEBUG] ip_layer_length : ", ip_layer_length)

        if (counter >= 2):
            test = False
        else:
            counter += 1

        # Full extraction after getting length

        end_range = ip_layer_length*8 #ip_layer_length*8 # - LOOK_BACK_PACKET
        decompressor.decompress(key_array=key_array,
                                start_range=start_range, 
                                end_range=end_range)

        data_decompress_bit_padded = decompressor.data_decompress
        print("[DEBUG] data_decompress_bit_padded : ", data_decompress_bit_padded)

        ## Remove pading
        data_decompress_bit = \
            data_decompress_bit_padded[LOOK_BACK_PACKET:]

        print("[DEBUG] decompressor.data_decompress LEN : ", len(decompressor.data_decompress))
        data_decompress_int = transform_packet_bit_int(
                array=np.array(decompressor.data_decompress), size=8)
        print("[DEBUG] dec.data_decompress LEN : ", data_decompress_int)

        
        # Update the context
        context_mapper.update(key_array=key_array,
                              data=data_decompress_bit)

        # Reset decompressor when compression is over
        decompressor.reset()
        
       

   
