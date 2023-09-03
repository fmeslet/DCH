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

FILENAME = "COAP.pcapng"
CHANGE_IPV6 = True

DATA_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/DATA_RAW/"
DATA_PATH = DATA_DIR + FILENAME
SAVE_DIR = "/users/rezia/fmesletm/HEADER_COMPRESSION/DATA/"



def array_bin_to_bytes(x):
    """Convert binary values to bytes.

    Args:
        x (np.array): array of binary value (0 or 1).

    Returns:
        np.array: array with bytes values.
    """
    x_int = array_bin_to_int(x)
    x_bytes = array_int_to_bytes(x_int)
    x_new = b"".join(x_bytes)
    return x_new


def array_bin_to_int(x, size=8):
    """Convert array of binary values (0 or 1)
    to integer values.

    Args:
        x (np.array): data to convert.
        size (int, optional): number of binary 
        value to agregate. Defaults to 8.

    Returns:
        np.array: array converted.
    """
    x_new = []    
    for i in range(
            0, x.size, size):
        value = 0
        for k, j in enumerate(reversed(range(size))):
            #print("i: ", i, " // ", "j: ", j)
            value += x[i+k]*(2**j)
            #print("value: ", value)
            
        x_new.append(value)
    x_new = np.array(x_new)
    return x_new


def int_to_bin(x, size=8):
    """Convert an integer to a binary string.

    Args:
        x (int): integer value to convert.
        size (int, optional): number of bits used. Defaults to 8.

    Returns:
        str: the binary value corresponding to the integer.
    """
    val = str(bin(x))[2:]
    s = len(val)
    gap = size - s
    
    if (gap < 0):
        raise "Error size limit too short"
        #size = max(s, size)
    
    gap_val = '0'*gap
    return gap_val+val


def array_hex_to_bin(x, size=8):
    """Convert array with hexadecimal values to binary
    values (0 or 1).

    Args:
        x (np.array): array to convert.
        size (int, optional): number of 
        binary values to used. Defaults to 8.

    Returns:
        np.array: array converted to binary values.
    """
    # Convert to int
    x_int = np.array(
        [int(i, 16) for i in x])
    
    # Convert to bin
    x_new = []
    
    for i in range(len(x_int)):
        x_bin = int_to_bin(
            x=x_int[i], size=size) 
        x_new.extend(x_bin)
        
    x_new = np.array(
        list(x_new), dtype=np.uint8)
    return x_new


def array_int_to_bin(x, size=8):
    """Convert array of integer into binary
    format (0 or 1).

    Args:
        x (np.array): array to convert.
        size (int, optional):  number of 
        binary values to used. Defaults to 8.

    Returns:
        np.array: array converted to binary values.
    """
    x_new = []
    for v in x:
        x_new_tmp = int_to_bin(
            x=v, size=size)
        x_new.extend(x_new_tmp)
        
    x_new = np.array(
        list(x_new), dtype=np.uint8)
    return x_new


def array_bin_to_hex(x, size=8):
    """Convert binary array (0 or 1) into
    hexadecimal array.

    Args:
        x (np.array): array to convert
        size (int, optional): number of 
        binary values to used. Defaults to 8.

    Returns:
        np.array: array converted to 
        haxadecimal values.
    """
    x_new = []
    
    for i in range(
            0, x.size, size):
        value = 0
        for k, j in zip(range(size), 
                        reversed(range(size))):
            value += x[i+k]*(2**j)
            
        x_new.append(hex(value))
    x_new = np.array(x_new)
    return x_new


def array_int_to_bytes(x, size=8):
    """Convert integer array into
    bytes array.

    Args:
        x (np.array): integer array.
        size (int, optional): size of the array. Defaults to 8.

    Returns:
        np.array: array converted to 
        bytes values.
    """
    x_new = [(int(x[i])).to_bytes(
            1, byteorder='big') for i in range(len(x))]
    return x_new


def array_bin_to_bytes(array_bin):
    """Convert binary array (0 or 1) into
    bytes array.

    Args:
        array_bin (np.array): binary array (0 or 1).

    Returns:
        np.array: array converted to 
        bytes values.
    """
    array_int = array_bin_to_int(
        array_bin)
    #print("array_int : ", array_int)
    array_bytes = array_int_to_bytes(
        array_int)
    array_bytes = b"".join(array_bytes)
    return array_bytes


def transform_packet_bytes_bit(packet_bytes, length=1522):
    """Convert packet in bytes to binaries (0 or 1).

    Args:
        array_bin (np.array): binary array (0 or 1).

    Returns:
        np.array: packet converted to 
        binary values.
    """
    payload_bits = ''.join(format(byte, '08b') for byte in packet_bytes)
    def split(bits):
        return [bit for bit in bits]
    packet_bit_array = np.array(split(payload_bits), dtype=np.int8)
    packet_bit_array_pad = np.lib.pad(packet_bit_array, 
                            (0,length-packet_bit_array.shape[0]), 
                            'constant', constant_values=(0))
    packet_bit_array_pad = np.reshape(packet_bit_array_pad, (length, 1))
    return packet_bit_array_pad.astype(np.int8)


class PcapProcessing():
    """Process a PCAP.
    """
    def __init__(self, lengths, 
                 mac_addr_prefix,
                 change_ipv6=False):
        """Constructor.

        Args:
            lengths (int): maximum length of packets.
            mac_addr_prefix (np.array): prefix for MAC address.
            convert_to_ipv6 (bool, optional): convert IPv4 
            address to IPv6. Defaults to False.
        """
        
        # Save variables
        self.df = pd.DataFrame()
        
        # Save data in int format
        self.lengths = lengths
        self.X = np.empty((0, self.lengths))
        self.X_reshape = None
        self.X_reshape_output = None
        
        # Init filename
        self.filename = ""
        
        # Convert Ipv4 to Ipv6
        self.change_ipv6 = change_ipv6
        self.mac_addr_prefix = mac_addr_prefix
            
        self.dict_ipv4_to_mac_addr = {} # Map Ipv4 address to mac address
        self.dict_mac_addr_to_ipv6 = {} # Map mac address to ipv6

        
    def reduce_memory_df(self, df):
        """Reduce DataFrame memory footprint.

        Args:
            df (pd.DataFrame): data about packets.

        Returns:
            pd.DataFrame: DataFrame with less memory 
            footprint.
        """
        # Set type for each columns
        return df

    
    def reduce_memory_npy(self, array):
        """Reduce Numpy array memory footprint.

        Args:
            array (np.array): array of packets.

        Returns:
            np.array: Numpy array with less memory
            footprint.
        """
        return array.astype(np.uint8)

    
    def add_df(self, feat_dict):
        """Add rows to DataFrame and reduce 
        his memory footprint.

        Args:
            feat_dict (dict): dictionary with 
            features to add. 
        """
        layers_df = pd.DataFrame(feat_dict, index=[0])
        #print(layers_df)
        layers_df_reduced = self.reduce_memory_df(layers_df)
        self.df = pd.concat([self.df, layers_df_reduced], axis=0)
        
        
    def compute_eui64_address(
        self, mac_address_hex):
        """Compute EUI64 address.

        Args:
            mac_address_hex (np.array): MAC address in an 
            array of hexadecimals.

        Returns:
            np.array: EUI64 address.
        """
        
        # Process first bit
        mac_address_first_bin = array_hex_to_bin(
            mac_address_hex[0:1], size=8)

        # Reverse second bit
        if (mac_address_first_bin[6] == 0):
            mac_address_first_bin[6] = 1
        else:
            mac_address_first_bin[6] = 0

        # Re set bit value to hex
        mac_address_first_hex = array_bin_to_hex(
            mac_address_first_bin, size=8)
        mac_address_hex[0] = mac_address_first_hex[0]

        # Fill with intermediate values
        eui64_addr_hex = np.concatenate(
            (mac_address_hex[:3], ["0xff", "0xfe"],
             mac_address_hex[-3:]))
        
        return eui64_addr_hex
        
        
    def compute_ipv6_addr(
        self, mac_addr):
        """Compute IPv6 link local address (FE80:...).

        Args:
            mac_addr (int): MAC address.

        Returns:
            np.array: IPv6 address in hexadecimals.
        """

        # Convert mac address to hex
        mac_addr_raw = mac_addr.split(":")
        mac_addr_hex = [str("0x"+mac_addr_raw[i]) for i in range(len(mac_addr_raw))]
        
        # Compute emui64
        emui64_addr_hex = self.compute_eui64_address(
            mac_addr_hex)
        
        # Build link-local address
        ipv6_addr = np.concatenate(
            (["0xfe", "0x80"],["0x00"]*6, 
             emui64_addr_hex))
        
        return ipv6_addr
        
        
    def convert_ipv6_addr(
        self, ipv4_addr):
        """Convert IPv4 address to IPv6.

        Args:
            ipv4_addr (np.array): IPv4 address 
            in hexadecimals.

        Returns:
            np.array: IPv6 address in hexadecimals.
        """
        
        # Check if address if already convert
        if (ipv4_addr in self.dict_ipv4_to_mac_addr):
            
            # Get mac and next ipv6 address
            mac_addr = self.dict_ipv4_to_mac_addr[
                ipv4_addr]
            ipv6_addr = self.dict_mac_addr_to_ipv6[
                mac_addr]
            
        else:
        
            # Generate mac address
            mac_addr = np.random.randint(0, 255, 3)
            if (self.mac_addr_prefix is None):
                mac_addr_prefix = np.random.randint(
                    0, 255, 3)
                mac_addr = np.concatenate(
                    (mac_addr_prefix, mac_addr))
            else:
                mac_addr = np.concatenate(
                    (self.mac_addr_prefix, mac_addr))
            
            # Generate mac
            
            ## Convert to hex
            
            mac_addr = [hex(e).upper()[2:] for e in mac_addr]
            #print("[DEBUG][convert_ipv4_to_ipv6][else] mac_addr after hex(): ", mac_addr)
            for i, f in enumerate(mac_addr):
                if (len(f) == 1):
                    mac_addr[i] = '0'+f
            #print("[DEBUG][convert_ipv4_to_ipv6][else] mac_addr after hex() AFTER FOR: ", mac_addr)
            
            ## Join element with ':'
            
            mac_addr = ':'.join(mac_addr)
            #print("[DEBUG][convert_ipv4_to_ipv6][else] mac_addr after join: ", mac_addr)
            self.dict_ipv4_to_mac_addr[
                ipv4_addr] = mac_addr
            
            #print("[DEBUG][convert_ipv4_to_ipv6][else] mac_addr: ", mac_addr)

            # Generate ipv6 address
            ipv6_addr = self.compute_ipv6_addr(
                mac_addr)
            self.dict_mac_addr_to_ipv6[
                mac_addr] = ipv6_addr
            
            #print("[DEBUG][convert_ipv4_to_ipv6][else] ipv6_addr: ", ipv6_addr)
        
        return ipv6_addr
    
    
    def convert_ip_addr(
        self, pkt):
        """Convert IPv6 address to link local address.

        Args:
            pkt (scapy.layers): packet.

        Returns:
            np.array: IPv6 header in bytes.
        """
    
        layers = self.get_layers(packet=pkt)
        ipv6_headers = np.zeros((40*8,))
    
        # Convert addr
        ipv6_src_addr_hex = self.convert_ipv6_addr(
                pkt['IPv6'].src)
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_addr_src_hex: ", ipv6_src_addr_hex)
        ipv6_dst_addr_hex = self.convert_ipv6_addr(
                pkt['IPv6'].dst)
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_addr_dst_hex: ", ipv6_dst_addr_hex)
        
        # Convert to bin
        
        ipv6_src_addr_bit = array_hex_to_bin(
            ipv6_src_addr_hex)
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_src_addr_bit: ", 
        #      ipv6_src_addr_bit)
        ipv6_dst_addr_bit = array_hex_to_bin(
            ipv6_dst_addr_hex)
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_dst_addr_bit: ", 
        #      ipv6_dst_addr_bit)
        
        # Set version 6
        ipv6_headers[1:3] = 1
        
        #print("[DEBUG][convert_ipv4_to_ipv6](version) ipv6_headers: ", 
        #      ipv6_headers)

        # Set traffic class
        #ipv6_headers[4:12] = 0
        
        # Next header
        if ('TCP' in layers):
            next_header_bit = array_int_to_bin(
                [6], size=8)
        elif ('UDP' in layers):
            next_header_bit = array_int_to_bin(
                [17], size=8)
        elif ("ICMPv6RPL" in layers):
            next_header_bit = array_int_to_bin(
                [58], size=8)
        elif ("ICMPv6" in layers):
            next_header_bit = array_int_to_bin(
                [58], size=8)
            
        #print("[DEBUG][convert_ipv4_to_ipv6] next_header_bit: ", next_header_bit)
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_headers[48:55]: ", ipv6_headers[48:56])
        
        ipv6_headers[48:56] = next_header_bit
        
        #print("[DEBUG][convert_ipv4_to_ipv6](next_header_bits) ipv6_headers: ", 
        #      ipv6_headers)

        # Set flow label
        if ('TCP' in layers):
            src_port_bit = array_int_to_bin(
                [pkt['TCP'].sport], size=16)
            dst_port_bit = array_int_to_bin(
                [pkt['TCP'].dport], size=16)
        elif ('UDP' in layers):
            src_port_bit = array_int_to_bin(
                [pkt['UDP'].sport], size=16)
            dst_port_bit = array_int_to_bin(
                [pkt['UDP'].dport], size=16)
            
        if (('TCP' in layers) or 
            ('UDP' in layers)):
            flow_label_bit = self.compute_flow_label_bit(
                            ipv6_src_addr_bit.copy(), 
                            ipv6_dst_addr_bit.copy(),
                            src_port_bit.copy(), 
                            dst_port_bit.copy(),
                            next_header_bit.copy())
        else:
            flow_label_bit = np.zeros((20,))
        
        #print("[DEBUG][convert_ipv4_to_ipv6] flow_label_bit: ", 
        #      flow_label_bit)
        
        ipv6_headers[12:32] = flow_label_bit
        
        #print("[DEBUG][convert_ipv4_to_ipv6](flow_label_bit) ipv6_headers: ", 
        #      ipv6_headers)

        # Set payload length
        payload_length = pkt['IPv6'].plen
        payload_length_bit = array_int_to_bin(
            [payload_length], size=16)
        ipv6_headers[32:48] = payload_length_bit
        
        #print("[DEBUG][convert_ipv4_to_ipv6](payload_length_bit) ipv6_headers: ", 
        #      ipv6_headers)

        # Hop limit, eq. TTL
        hop_limit_bit = array_int_to_bin(
                [pkt['IPv6'].hlim], size=8)
        ipv6_headers[56:64] = hop_limit_bit

        # Source address
        ipv6_headers[64:192] = ipv6_src_addr_bit
        
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_src_addr_bit: ", 
        #      ipv6_src_addr_bit)
        #print("[DEBUG][convert_ipv4_to_ipv6](ipv6_src_addr_bit) ipv6_headers: ", 
        #      ipv6_headers)

        # Destination address
        ipv6_headers[192:320] = ipv6_dst_addr_bit
        
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_dst_addr_bit: ", 
        #      ipv6_dst_addr_bit)
        #print("[DEBUG][convert_ipv4_to_ipv6](ipv6_dst_addr_bit) ipv6_headers: ", 
        #      ipv6_headers)
        
        # Rebuild packet
        ipv6_headers_bytes = array_bin_to_bytes(
            ipv6_headers)
        
        #print("[DEBUG][convert_ipv4_to_ipv6] ipv6_headers_bytes: ", 
        #      ipv6_headers_bytes)
        
        return ipv6_headers_bytes
         
        
    def add_arithmetic_bit(
        self, array_a, array_b):
        
        if (len(array_b) < len(array_a)):
            limit_size = len(array_b)
            array_c = array_a
        else:
            limit_size = len(array_a)
            array_c = array_b
            
        array_c[:limit_size] = array_a[:limit_size] + \
                array_b[:limit_size]
        idx = np.where(array_c == 2)[0].astype(int)
        array_c[idx] = 0
        return array_c

    
    def compute_flow_label_bit(
        self, ipv6_src_address_bit, 
        ipv6_dst_address_bit,
        src_port_bit, dst_port_bit,
        next_header_bit):
        """Compute flow label field for IPv6 packet.

        Args:
            ipv6_src_address_bit (np.array): IPv6 source address in binary format.
            ipv6_dst_address_bit (np.array): IPv6 destination address in binary format.
            src_port_bit (np.array): source port in binary format.
            dst_port_bit (np.array): destination port in binary format.
            next_header_bit (np.array): next header field in binary format.

        Returns:
            np.array: flow label in binary format.
        """
        
        print("")
        
        # Arithmetic sum ADDR and next header
        both_ipv6_src_address_bit = self.add_arithmetic_bit(
            ipv6_src_address_bit[:64],
            ipv6_src_address_bit[64:])
        
        #print("[DEBUG][compute_flow_label_bit] both_ipv6_src_address_bit: ", both_ipv6_src_address_bit)
        
        both_ipv6_dst_address_bit = self.add_arithmetic_bit(
            ipv6_dst_address_bit[:64],
            ipv6_dst_address_bit[64:])
        
        #print("[DEBUG][compute_flow_label_bit] both_ipv6_src_address_bit: ", both_ipv6_dst_address_bit)
        
        both_ipv6_address_bit = self.add_arithmetic_bit(
            both_ipv6_src_address_bit,
            both_ipv6_dst_address_bit)
        
        #print("[DEBUG][compute_flow_label_bit] both_ipv6_address_bit: ", both_ipv6_address_bit)
        
        arithmetic_sum_bit = self.add_arithmetic_bit(
            both_ipv6_address_bit,
            next_header_bit)
        
        #print("[DEBUG][compute_flow_label_bit] arithmetic_sum_bit: ", arithmetic_sum_bit)
        
        # Apply Von Neumann algo
        i = 0
        arithmetic_sum_bit_flip = np.flip(
            arithmetic_sum_bit)
        
        #print("[DEBUG][compute_flow_label_bit] arithmetic_sum_bit_flip: ", arithmetic_sum_bit_flip)
        
        output_bit = []
        while ((i < (len(arithmetic_sum_bit)-1)) and
               (len(output_bit) < 16)):
            
            select_bit = arithmetic_sum_bit_flip[i:i+2]
            #print("[DEBUG][compute_flow_label_bit] select_bit: ", select_bit)
            select_bit = ''.join(
                select_bit.astype(str))  
            
            #print("[DEBUG][compute_flow_label_bit] select_bit: ", select_bit)
               
            if ((select_bit == '00') or 
                (select_bit == '11')):
                pass
            elif (select_bit == '01'):
                output_bit.append(0)
            elif (select_bit == '10'):
                output_bit.append(1)
            
            i = i + 2
            
        #print("[DEBUG][compute_flow_label_bit] output_bit: ", output_bit)
               
        # Add the two port number and the result
        ipv6_port_bit = self.add_arithmetic_bit(
            src_port_bit, dst_port_bit)
        #print("[DEBUG][compute_flow_label_bit] ipv6_port_bit: ", ipv6_port_bit)
        arithmetic_sum_bit = self.add_arithmetic_bit(
            ipv6_port_bit, output_bit)
        #print("[DEBUG][compute_flow_label_bit] arithmetic_sum_bit: ", arithmetic_sum_bit)
               
        # Shift four bit left
        flow_label_bit = np.concatenate(
            (arithmetic_sum_bit, [0, 0, 0, 0]))
        #print("[DEBUG][compute_flow_label_bit] flow_label_bit: ", flow_label_bit)
               
        # Unlikely event
        if (sum(flow_label_bit) == 0):
            flow_label_bit[0] = 1
        
        return flow_label_bit
        
    
    def save_df(self, counter):
        """Save DataFrame.

        Args:
            counter (int): value associated to the packet.
        """
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(f"{SAVE_DIR}csv/df_{counter}_{self.filename}.csv", index=False)
        self.df = pd.DataFrame()
        gc.collect()
        print(f"DataFrame {counter} saved, file {self.filename} !")

        
    def add_npy(self, packet, feat_dict):
        """Concatenate new packet to numpy array.

        Args:
            packet (scapy.packet): packet in bytes format.
            feat_dict (dict): features to plot for exception.

        Raises:
            e: exception in case of error during the concatenation.
        """
        # Transform to bit + pad
        packet_bytes = bytes_encode(packet)
        print("[DEBUG][add_npy] packet_bytes: ", packet_bytes)
        
        try:
            packet_bit = transform_packet_bytes_bit(
              packet_bytes=packet_bytes, length=self.lengths)
            packet_bit = np.reshape(packet_bit, (1, self.lengths))
            packet_bit_reduced = self.reduce_memory_npy(packet_bit)
        except Exception as e:
            print(feat_dict)
            raise e
            
        self.X = np.concatenate(
            (self.X, packet_bit_reduced), axis=0)

        
    def save_npy(self, counter):
        """Save numpy array to disk.

        Args:
            counter (int): counter associated to the packet.
        """
        # Transform to bit + pad
        np.save(arr=self.X, file=f"{SAVE_DIR}npy/arr_{counter}_{self.filename}_bit.npy")
        self.X = np.empty((0, self.lengths))
        gc.collect()
        print(f"Array {counter} saved, file {self.filename} !")
        
        
    def get_layers(self, packet):
        """Get layers ascociated to the packet.

        Args:
            packet (scapy.layers): packet.

        Returns:
            list: layers name.
        """
        layer = []
        for i in packet.layers():
            name = str(i).split('.')[-1][:-2]
            layer.append(name)
        return layer

    
    def init_iterateur(self, iterator, value):
        """Init iterator.

        Args:
            iterator (scapy.all.PcapReader): _description_
            value (int): counter associated to packet.
        """
        i = 0
        while (i < value):
            iterator.next()
            i += 1

            
    def extract_features(self, packet, num_packet):
        """Extract the features of a packet.

        Args:
            packet (scapy.packet): packet.
            num_packet (int): counter associated to each packet.

        Returns:
            dict: dictionnary with each feature extracted from the packet.
        """
        feat_dict = {}

        # Extract layers
        layers = self.get_layers(packet=packet)
        for i in range(len(layers)):
            if(i >= len(layers)):
                feat_dict[f'layers_{i}'] = None
                feat_dict[f'length_{i}'] = int(0)
            else:
                feat_dict[f'layers_{i}'] = layers[i]
                feat_dict[f'length_{i}'] = int(len(packet[layers[i]]))

        # Extract timestamps
        feat_dict['timestamps'] = float(packet.time)

        # Extract total length
        feat_dict['length_total'] = int(len(packet))

        # Extract MAC address
        if ('Ether' in layers):
            feat_dict['mac_src'] = packet['Ethernet'].src
            feat_dict['mac_dst'] = packet['Ethernet'].dst
        elif ('Dot3' in layers):
            feat_dict['mac_src'] = packet['Dot3'].src
            feat_dict['mac_dst'] = packet['Dot3'].dst
        else:
            feat_dict['mac_src'] = None
            feat_dict['mac_dst'] = None
            
        # Extract IP address
        if ('IP' in layers):
            feat_dict['ip_src'] = packet['IP'].src
            feat_dict['ip_dst'] = packet['IP'].dst
        else:
            feat_dict['ip_src'] = None
            feat_dict['ip_dst'] = None

        # Extract flags if TCP and ports
        if ('TCP' in layers):
            feat_dict['flags'] = packet['TCP'].flags.value
            feat_dict['sport'] = packet['TCP'].sport
            feat_dict['dport'] = packet['TCP'].dport
            
            cond_mqtt = ((feat_dict['sport'] == 1883) | 
                         (feat_dict['dport'] == 1883))
            if (cond_mqtt):
                feat_dict['application'] = "MQTT"
            else:
                feat_dict['application'] = "OTHERS"
            
        elif ('UDP' in layers):
            feat_dict['flags'] = None
            feat_dict['sport'] = packet['UDP'].sport
            feat_dict['dport'] = packet['UDP'].dport
            
            cond_coap = ((feat_dict['sport'] == 5683) | 
                         (feat_dict['dport'] == 5683))
            if (cond_coap):
                feat_dict['application'] = "COAP"
            else:
                feat_dict['application'] = "OTHERS"
            
        else:
            feat_dict['flags'] = None
            feat_dict['sport'] = None
            feat_dict['dport'] = None
            feat_dict['application'] = "OTHERS"
        

        feat_dict['filename'] = self.filename
        feat_dict['num_packet'] = num_packet
        
        return feat_dict


    def fit(self, filename="", 
            start=0, inter=1):
        """Process a range of packet in a PCAP file.

        Args:
            filename (str, optional): name of file to process. Defaults to "".
            start (int, optional): start index. Defaults to 0.
            inter (int, optional): end index. Defaults to 1.
        """
        
        self.filename = filename
        iterator = PcapReader(DATA_PATH)
        i = start # Packet counter
        j = 0 # Counter for saving

        # if(end is None):
        #   end = 9e9

        # Init iterateur to start value
        self.init_iterateur(
            iterator=iterator, value=start)
        
        print("Init done !")

        # while ((i < num_packets) and (i < end)):
        for pkt in iterator:
            # Extract headers + save to df
            #pkt = iterator.next()
            
            # Get layers
            layers = self.get_layers(
                packet=pkt)
            print("[DEBUG][fit] layers: ", layers)

            # Extract dict
            feat_dict = self.extract_features(
                pkt, num_packet=i)
            
            # IP layers
            if ('IPv6' in layers):
                
                if (self.change_ipv6):
                    
                    ## Convert addr
                    ipv6_src_addr_hex = self.convert_ipv6_addr(
                            pkt['IPv6'].src)
                    ipv6_dst_addr_hex = self.convert_ipv6_addr(
                            pkt['IPv6'].dst)
                    
                    #print("[DEBUG][fit] ipv6_src_addr_hex: ", ipv6_src_addr_hex)
                    #print("[DEBUG][fit] ipv6_dst_addr_hex: ", ipv6_dst_addr_hex)
                    
                    # Update MAC addr
                    feat_dict['mac_src_custom'] = self.dict_ipv4_to_mac_addr[
                                pkt['IPv6'].src]
                    feat_dict['mac_dst_custom'] = self.dict_ipv4_to_mac_addr[
                                pkt['IPv6'].dst]
                    
                    # Convert to byte
                    pkt = bytes_encode(pkt['IPv6'])
                    
                    #print("[DEBUG] pkt: ", pkt)
                    #print("[DEBUG] IPv6(pkt): ", IPv6(pkt).show())
                    #print("[DEBUG] AFTER IPv6(pkt): ", IPv6(pkt).show())
        
                    ## Convert to bin
                    #pkt_bit = array_hex_to_bin(pkt)
                    pkt_bit = transform_packet_bytes_bit(
                        pkt, length=self.lengths).ravel()
                    ipv6_src_addr_bit = array_hex_to_bin(
                        ipv6_src_addr_hex)
                    ipv6_dst_addr_bit = array_hex_to_bin(
                        ipv6_dst_addr_hex)
                    
                    #print("[DEBUG][fit] pkt_bit: ", pkt_bit.ravel())
                    
                    # Set value to bit
                    pkt_bit[64:192] = ipv6_src_addr_bit
                    pkt_bit[192:320] = ipv6_dst_addr_bit
                    
                    #print("[DEBUG][fit] ipv6_src_addr_bit: ", ipv6_src_addr_bit)
                    #print("[DEBUG][fit] ipv6_dst_addr_bit: ", ipv6_dst_addr_bit)
                    
                    # Address to bytes
                    pkt = array_bin_to_bytes(
                            pkt_bit)
                    
                    #print("[DEBUG] AFTER IPv6(pkt): ", IPv6(pkt).show())
                    
                else:
                    
                    pkt = bytes_encode(pkt['IPv6'])
                
                
            #print(IP(ipv6_headers_bytes))

            # Remove payloads
            # self.remove_payload(packet=pkt, layers=self.get_layers(pkt))

            # # Concat to X and df
            self.add_npy(packet=pkt, 
                         feat_dict=feat_dict)
            self.add_df(feat_dict=feat_dict)

            # On incrémente le compteur
            i += 1
            j += 1

            if(j == inter):
                self.save_df(counter=i)
                self.save_npy(counter=i)
                j = 0

        # Save the last data
        self.save_df(counter=i)
        self.save_npy(counter=i)

    # Preprocessing à faire après avoir extrait les données !
    def remove_payload(self, packet, layers):
        """Remove payload for each packet.

        Args:
            packet (scapy.layers): packet.
            layers (list): names of each layer in the packet. 
        """
        if ('Raw' in layers):
            packet[layers[-2]].remove_payload()


# MQTT Kaggle (length=110, length_max=105)
# MQTT IEEE (length=1365, length_max=1358)

pcap_processing = PcapProcessing(
                lengths=300*8, 
                mac_addr_prefix=[80, 2, 145], # eq. TTGO 50:02:91:9c:88:8c
                change_ipv6=CHANGE_IPV6)


pcap_processing.fit(
    filename=FILENAME, 
    start=0, inter=2000)

print("[DEBUG] pcap_processing.dict_mac_addr_to_ipv6: ", 
        pcap_processing.dict_mac_addr_to_ipv6)

print("")

print("[DEBUG] pcap_processing.dict_ipv4_to_mac_addr: ", 
        pcap_processing.dict_ipv4_to_mac_addr)
