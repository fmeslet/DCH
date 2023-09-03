#!/home/fmeslet/anaconda3/bin/python3.8
#-*-coding: utf-8 -*-

import numpy as np

# Prime number to compute hash
A = 54059 # a prime
B = 76963 # another prime
C = 86969 # yet another prime
FIRSTH = 37 # also prime

class ContextMapper():

    def __init__(
        self, size_context, 
        size_element):

        self.size_context = size_context    
        self.size_element = size_element

        self.size_context_ravel = self.size_context*\
                self.size_element

        self.data_context = {}
        self.context_size = {}


    # Same implementation as C++ Arduino
    #def compute_hash(self, key_value):
    #    a = 63689
    #    b = 378551
    #    h = 0

    #    for i in range(len(key_value)):
    #        h = h * a + key_value[i])
    #        a = a * b

    #    return h


    def compute_hash(
        self, key_array):

        h = hash(''.join(
            [str(k) for k in key_array])) % 256

        return h


    def update_key(self, 
                  key_value, 
                  data):

        # Extract array with key
        context_data = self.data_context[
                key_value];


        # Shift context data
        for i in range(self.size_context-1):
            for j in range(self.size_element): 
              context_data[(i*self.size_element) + j] = \
                    context_data[(i+1)*self.size_element+j]

        # Update last values of context
        for j in range(len(data)): 
            context_data[
                (self.size_context-1)*self.size_element + j] = data[j]

        # Fill the other part with 0
        for j in range(len(data), self.size_element): 
            context_data[(self.size_context-1)*self.size_element+j] = 0


        # Add to map
        self.data_context[key_value] = context_data
        self.context_size[key_value] = min(
            self.context_size[key_value]+1, 
            self.size_context)

        print("[DEBUG][update_key] key_value: ", key_value)
        print("[DEBUG][update_key] context_data: ", context_data)
  

    def update(self, 
               key_array,
               data): # Check if array is well sized

        # Compute key
        key_value = self.compute_hash(
            key_array)

        # Check if the key exist
        if (key_value in self.data_context):
            #pass
            self.update_key(
                key_value, data)
        else:
            self.create_key(key_value)


    def create_key(
        self, key_value):

        # Init context to zero
        context_data = np.zeros(
            (self.size_context*self.size_element))

        # Add to map
        self.data_context[key_value] = context_data
        self.context_size[key_value] = 0


    def get_data_context(
        self, key_array):

      # Compute key
      key_value = self.compute_hash(
        key_array)

      # Check if the key exist
      if (key_value in self.data_context):
        #print("[DEBUG][ContextMapper::getDataContext] keyValue %d exist !\n", keyValue);
        return self.data_context[key_value]
      else:
        #print("[DEBUG][ContextMapper::getDataContext] keyValue %d not EXIST !\n", keyValue);
        self.create_key(key_value)
        return self.data_context[key_value]


    def get_context_size(
        self, key_array):

        # Compute key
        key_value = self.compute_hash(
            key_array)

        if (key_value in self.context_size):
            return self.context_size[key_value]
        else:
            self.context_size[key_value] = 0
            return self.context_size[key_value]

    def get_size_element(self):
        return self.size_element

    def get_size_element(self):
        return self.size_context
        

