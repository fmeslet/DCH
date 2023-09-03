#ifndef ARITHMETIC_CODER_H
#define ARITHMETIC_CODER_H

#include <Arduino.h>



/* 
 * Encodes symbols and writes to an arithmetic-coded bit stream.
 */
class ArithmeticEncoder {
  
  /*---- Fields ----*/
  
  // Number of saved underflow bits. This value can grow without bound,
  // so a truly correct implementation would use a bigint.
  private: unsigned long numUnderflow;

  // Array of data compress and position of last bit
  private: uint8_t dataCompressBin[1500];
  private: int sizeDataCompressBin;
  private: uint8_t dataCompressHex[500];
  private: int sizeDataCompressHex;

  private: int numStateBits;
  
  // Maximum range (high+1-low) during coding (trivial), which is 2^numStateBits = 1000...000.
  private: uint64_t fullRange;
  
  // The top bit at width numStateBits, which is 0100...000.
  private: uint64_t halfRange;
  
  // The second highest bit at width numStateBits, which is 0010...000. This is zero when numStateBits=1.
  private: uint64_t quarterRange;
  
  // Minimum range (high+1-low) during coding (non-trivial), which is 0010...010.
  private: uint64_t minimumRange;
  
  // Maximum allowed total from a frequency table at all times during coding.
  private: uint64_t maximumTotal;
  
  // Bit mask of numStateBits ones, which is 0111...111.
  private: uint64_t stateMask;
  
  
  /*---- State fields ----*/
  
  // Low end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 0s.
  public: uint64_t low;
  
  // High end of this arithmetic coder's current range. Conceptually has an infinite number of trailing 1s.
  private: uint64_t high;
  
  
  /*---- Constructor ----*/
  
  // Constructs an arithmetic coding encoder based on the given bit output stream.
  public: explicit ArithmeticEncoder(int numBits);
  
  
  /*---- Methods ----*/
  
  // Encodes the given symbol based on the given frequency table.
  // Also updates this arithmetic coder's state and may write out some bits.
  public: void write(uint64_t* cumul, 
                     uint8_t symbol, 
                     int alphabetSize);


  private: void update(uint64_t* cumul, 
                       uint8_t symbol,
                       int alphabetSize);
  
  private: void shift();
  
  
  private: void underflow();

   
  public: void finish();


  // Getter
  public: uint8_t* getDataCompressBin();
  public: int getSizeDataCompressBin();
  public: uint8_t* getDataCompressHex();
  public: int getSizeDataCompressHex();
  
};


#endif
