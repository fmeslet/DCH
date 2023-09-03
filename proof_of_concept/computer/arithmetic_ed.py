#!/home/fmeslet/anaconda3/bin/python3.8
#-*-coding: utf-8 -*-

# ---- Arithmetic coding core classes ----

# Provides the state and behaviors that arithmetic coding encoders and decoders share.
class ArithmeticCoderBase(object):
    
    # Constructs an arithmetic coder, which initializes the code range.
    def __init__(self, statesize):
        
        # -- Configuration fields --
        # Number of bits for the 'low' and 'high' state variables. Must be at least 1.
        # - Larger values are generally better - they allow a larger maximum frequency total (MAX_TOTAL),
        #   and they reduce the approximation error inherent in adapting fractions to integers;
        #   both effects reduce the data encoding loss and asymptotically approach the efficiency
        #   of arithmetic coding using exact fractions.
        # - But larger state sizes increase the computation time for integer arithmetic,
        #   and compression gains beyond ~30 bits essentially zero in real-world applications.
        # - Python has native bigint arithmetic, so there is no upper limit to the state size.
        #   For Java and C++ where using native machine-sized integers makes the most sense,
        #   they have a recommended value of STATE_SIZE=32 as the most versatile setting.
        self.STATE_SIZE = statesize
        # Maximum range (high+1-low) during coding (trivial), which is 2^STATE_SIZE = 1000...000.
        self.MAX_RANGE = 1 << self.STATE_SIZE
        # Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
        self.MIN_RANGE = (self.MAX_RANGE >> 2) + 2
        # Maximum allowed total from a frequency table at all times during coding. This differs from Java
        # and C++ because Python's native bigint avoids constraining the size of intermediate computations.
        self.MAX_TOTAL = self.MIN_RANGE
        # Bit mask of STATE_SIZE ones, which is 0111...111.
        self.MASK = self.MAX_RANGE - 1
        # The top bit at width STATE_SIZE, which is 0100...000.
        self.TOP_MASK = self.MAX_RANGE >> 1
        # The second highest bit at width STATE_SIZE, which is 0010...000. This is zero when STATE_SIZE=1.
        self.SECOND_MASK = self.TOP_MASK >> 1
        
        # -- State fields --
        # Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
        self.low = 0
        # High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
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
    
    # Called to handle the situation when the top bit of 'low' and 'high' are equal.
    def shift(self):
        raise NotImplementedError()
    
    
    # Called to handle the situation when low=01(...) and high=10(...).
    def underflow(self):
        raise NotImplementedError()
        

# Encodes symbols and writes to an arithmetic-coded bit stream.
class ArithmeticEncoder(ArithmeticCoderBase):
    
    # Constructs an arithmetic coding encoder based on the given bit output stream.
    def __init__(self, statesize, bitout,
                 write_mode=False):
        super(ArithmeticEncoder, self).__init__(statesize)
        # The underlying bit output stream.
        self.output = bitout
        self.write_mode = write_mode
        # Number of saved underflow bits. This value can grow without bound.
        self.num_underflow = 0
        self.data_compress = []
    
    
    # Encodes the given symbol based on the given frequency table.
    # This updates this arithmetic coder's state and may write out some bits.
    def write(self, cumul, symbol):
        self.update(cumul, symbol)
    
    # Terminates the arithmetic coding by flushing any buffered bits, so that the output can be decoded properly.
    # It is important that this method must be called at the end of the each encoding process.
    # Note that this method merely writes data to the underlying output stream but does not close it.
    def finish(self):
        if (self.write_mode):
            self.output.write(1)
        else:
            self.data_compress.append(1)
            
    
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


# Reads from an arithmetic-coded bit stream and decodes symbols.
class ArithmeticDecoder(ArithmeticCoderBase):
    
    # Constructs an arithmetic coding decoder based on the
    # given bit input stream, and fills the code bits.
    def __init__(self, statesize, bitin, 
                 data_compress, 
                 write_mode=False):
        super(ArithmeticDecoder, self).__init__(statesize)
        # The underlying bit input stream.
        self.input = bitin
        self.data_compress = data_compress
        self.data_decompress = []
        self.rank = 0
        self.write_mode = write_mode
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        self.temp_read = []
        self.temp = []
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()


    def reset(self):
        self.data_decompress = []
        self.rank = 0
        # The current raw code bits being buffered, which is always in the range [low, high].
        self.code = 0
        self.temp_read = []
        self.temp = []
        for _ in range(self.STATE_SIZE):
            self.code = self.code << 1 | self.read_code_bit()
        
    
    # Decodes the next symbol based on the given frequency table and returns it.
    # Also updates this arithmetic coder's state and may read in some bits.
    def read(self, cumul, alphabet_size):
        
        # Translate from coding range scale to frequency table scale
        total = cumul[-1].item()
        range_value = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // range_value
        
        # A kind of binary search. Find highest symbol such that freqs.get_low(symbol) <= value.
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
        #print("symbol : ", symbol)
        self.data_decompress.append(symbol)
        #print("[DEBUG][read] self.data_decompress : ", self.data_decompress)
        #self.rank += 1
        return symbol
    
    
    def shift(self):
        self.code = ((self.code << 1) & self.MASK) | self.read_code_bit()
        #print("[DEBUG][shift] self.code : ", self.code)
        #self.rank += 1
    
    
    def underflow(self):
        self.code = (self.code & self.TOP_MASK) | ((self.code << 1) & (self.MASK >> 1)) | self.read_code_bit()
        #print("[DEBUG][underflow] self.code : ", self.code)
    
    
    # Returns the next bit (0 or 1) from the input stream. The end
    # of stream is treated as an infinite number of trailing zeros.
    def read_code_bit(self):
        #temp_read = self.input.read()
        #self.temp_read.append(temp_read)
        #print("[DEBUG][read_code_bit][write_mode=False] temp_read : ", temp_read)
        if (self.write_mode):
            temp = self.input.read()
            #print("[DEBUG][read_code_bit][write_mode=False] temp : ", temp)
        else:
            try:
                temp = self.data_compress[self.rank]
                #print("[DEBUG][read_code_bit][write_mode=False] temp : ", temp)
                #print("[DEBUG][read_code_bit][write_mode=False] self.rank : ", self.rank)
            except Exception as e:
                temp = -1
            #print("self.rank : ", self.rank)
            #print("temp : ", temp)
            self.rank += 1
            
        self.temp.append(temp)
            
        if temp == -1:
            temp = 0
            
        #self.data_decompress.append(temp)
        return temp


# ---- Bit-oriented I/O streams ----

# A stream of bits that can be read. Because they come from an underlying byte stream,
# the total number of bits is always a multiple of 8. The bits are read in big endian.
class BitInputStream(object):
    
    # Constructs a bit input stream based on the given byte input stream.
    def __init__(self, inp):
        # The underlying byte stream to read from
        self.input = inp
        # Either in the range [0x00, 0xFF] if bits are available, or -1 if end of stream is reached
        self.currentbyte = 0
        # Number of remaining bits in the current byte, always between 0 and 7 (inclusive)
        self.numbitsremaining = 0
    
    
    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or -1 if
    # the end of stream is reached. The end of stream always occurs on a byte boundary.
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
    
    
    # Reads a bit from this stream. Returns 0 or 1 if a bit is available, or raises an EOFError
    # if the end of stream is reached. The end of stream always occurs on a byte boundary.
    def read_no_eof(self):
        result = self.read()
        if result != -1:
            return result
        else:
            raise EOFError()
    
    
    # Closes this stream and the underlying input stream.
    def close(self):
        self.input.close()
        self.currentbyte = -1
        self.numbitsremaining = 0


# A stream where bits can be written to. Because they are written to an underlying
# byte stream, the end of the stream is padded with 0's up to a multiple of 8 bits.
# The bits are written in big endian.
class BitOutputStream(object):
    
    # Constructs a bit output stream based on the given byte output stream.
    def __init__(self, out):
        self.output = out  # The underlying byte stream to write to
        self.currentbyte = 0  # The accumulated bits for the current byte, always in the range [0x00, 0xFF]
        self.numbitsfilled = 0  # Number of accumulated bits in the current byte, always between 0 and 7 (inclusive)
    
    
    # Writes a bit to the stream. The given bit must be 0 or 1.
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
    
    
    # Closes this stream and the underlying output stream. If called when this
    # bit stream is not at a byte boundary, then the minimum number of "0" bits
    # (between 0 and 7 of them) are written as padding to reach the next byte boundary.
    def close(self):
        while self.numbitsfilled != 0:
            self.write(0)
        self.output.close()
