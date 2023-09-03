#!/home/fmeslet/anaconda3/bin/python3.8
#-*-coding: utf-8 -*-

import numpy as np

from arithmetic_ed import *

class Decompressor():

    def __init__(self,
                 alphabet_size,
                 context_mapper,
                 frequency_table,
                 overhead,
                 bitin=None,
                 write_mode=False):

        self.alphabet_size = alphabet_size
        self.context_mapper = context_mapper
        self.write_mode = write_mode
        self.frequency_table = frequency_table
        self.overhead = overhead

        self.dec = None

        self.reset()


    def reset(self):
        
        self.data_decompress = []

        prob = np.ones(
            self.alphabet_size)/self.alphabet_size
        self.cumul = np.zeros(
            self.alphabet_size+1, 
            dtype=np.uint64)
        self.cumul[1:] = np.cumsum(
            np.around(prob*10000000.) + 1)


    def fit(self, 
             data_compress):

        # We fix to 32 bit !
        self.dec = ArithmeticDecoder(
            32, bitin=None, # 16
            data_compress=data_compress, 
            write_mode=self.write_mode)

        
    def decompress(self, 
                   key_array,
                   #data_compress,
                   start_range, 
                   end_range):

        #print("")

        context_size = self.context_mapper.get_context_size(
                    key_array)
        data_context = self.context_mapper.get_data_context(
                key_array) # Format binary

        #self.dec.data_compress = data_compress

        #print("[DEBUG][decompress] context_size: ", context_size)
        #print("[DEBUG][decompress] data_context: ", data_context)
        #print("[DEBUG][decompress] data_compress: ", data_compress)

        #print("[DEBUG][decompress] self.dec.data_compress: ", 
        #        self.dec.data_compress)
        #print("[DEBUG] self.overhead: ", self.overhead)

        # Set overhead
        for k in range(
           start_range, self.overhead):
            symbol = self.dec.read(
                self.cumul, self.alphabet_size)
            self.data_decompress.append(symbol)

            #print("[DEBUG][decompress] self.data_decompress: ", self.data_decompress)

            #self.dec.read(
            #    self.cumul, self.alphabet_size)
            #self.data_decompress.append(symbol)
            #print("[DEBUG][decompress] self.dec.data_decompress: ", self.dec.data_decompress)

        #print("[DEBUG][decompress] start_range: ", start_range)
        #print("[DEBUG][decompress] end_range: ", end_range)
        #print("[DEBUG][decompress] self.dec.data_decompress: ", 
        #        self.dec.data_compress)

        # Set decompression
        for i in range(
            start_range, end_range):

            # Pré-processing
            symbols_decompress = np.array(
                self.data_decompress)[-self.overhead:] #.reshape((1, -1))
            #print("[DEBUG][decompress] symbols_decompress: ", symbols_decompress)
            data = np.concatenate((data_context, 
                                   symbols_decompress), axis=-1)
            #print("[DEBUG][decompression_packet] data shape : ", data.shape)
            data = data.reshape((1, -1))
            #print("[DEBUG][decompression_packet] data shape : ", data.shape)

            # Predict
            pos = np.array([i])
            ctx = [context_size]
            #print("[DEBUG][decompression_packet] i : ", i)
            #print("[DEBUG][decompression_packet] ctx : ", ctx)
            prob = self.frequency_table.predict(data=data,
                                                pos=pos,
                                                ctx=ctx, 
                                                soft=True)
            #print("[DEBUG][decompress] prob : ", prob)
             
            # Update cumsum and read
            self.cumul[1:] = np.cumsum(
                np.around(prob*10000000.) + 1) # 10000
            print("[DEBUG][decompress] self.cumul: ", self.cumul)
            symbol = self.dec.read(self.cumul, 
                                   self.alphabet_size)
            print("[DEBUG][decompress] symbol: ", symbol)
            self.data_decompress.append(symbol)

            #print("[DEBUG][decompress] self.dec.data_decompress: ", 
            #    self.dec.data_decompress)

        
