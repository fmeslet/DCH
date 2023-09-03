#include "arithmetic_coder.h"

//#include <limits>
//#include <stdexcept>


/*
 * ARITHMETIC ENCODER
 */


ArithmeticEncoder::ArithmeticEncoder(int numBits):
  numUnderflow(0)
{

  numStateBits = numBits;
  fullRange = static_cast<decltype(fullRange)>(1) << numStateBits;
  halfRange = fullRange >> 1;  // Non-zero
  quarterRange = halfRange >> 1;  // Can be zero
  minimumRange = quarterRange + 2;  // At least 2
  maximumTotal = std::min(std::numeric_limits<decltype(fullRange)>::max() / fullRange, minimumRange);
  stateMask = fullRange - 1;
  low = 0;
  high = stateMask;
  
  sizeDataCompressBin = sizeof(
    dataCompressBin)/sizeof(dataCompressBin[0]);
  for (int i=0; i<sizeDataCompressBin; i++)
  {
    dataCompressBin[i] = 0;
  }
  sizeDataCompressBin = 0; // Reset for next increasing

  sizeDataCompressHex = sizeof(
    dataCompressHex)/sizeof(dataCompressHex[0]);
  for (int i=0; i<sizeDataCompressHex; i++)
  {
    dataCompressHex[i] = 0;
  }
  sizeDataCompressHex = 0;
  
}


void ArithmeticEncoder::update(uint64_t* cumul, 
                               uint8_t symbol,
                               int alphabetSize) {

  fullRange = static_cast<decltype(fullRange)>(1) << numStateBits;
  halfRange = fullRange >> 1;  // Non-zero
  quarterRange = halfRange >> 1;  // Can be zero
  minimumRange = quarterRange + 2;  // At least 2
  maximumTotal = std::min(std::numeric_limits<decltype(fullRange)>::max() / fullRange, minimumRange);
  stateMask = fullRange - 1;
                                  
  uint64_t range = high - low + 1;
  
  // Frequency table values check
  uint64_t total = cumul[alphabetSize];
  uint64_t symLow = cumul[symbol];
  uint64_t symHigh = cumul[symbol+1];
  
  // Update range
  uint64_t newLow  = low + symLow  * range / total;
  uint64_t newHigh = low + symHigh * range / total - 1;
  low = newLow;
  high = newHigh;
  
  // While low and high have the same top bit value, shift them out
  while (((low ^ high) & halfRange) == 0) {
    shift();
    low  = ((low  << 1) & stateMask);
    high = ((high << 1) & stateMask) | 1;
  }
  
  // While low's top two bits are 01 and high's are 10, delete the second highest bit of both
  while ((low & ~high & quarterRange) != 0) {
    underflow();
    low = (low << 1) ^ halfRange;
    high = ((high ^ halfRange) << 1) | halfRange | 1;
  }
  
}


void ArithmeticEncoder::write(uint64_t* cumul, uint8_t symbol, int alphabetSize) {
  this->update(cumul, symbol, alphabetSize);
}


void ArithmeticEncoder::shift() {
  int bit = static_cast<int>(low >> (numStateBits - 1));
  dataCompressBin[sizeDataCompressBin] = bit;
  sizeDataCompressBin = sizeDataCompressBin + 1;

  // Write out the saved underflow bits
  for (; numUnderflow > 0; numUnderflow--) 
  {
    dataCompressBin[sizeDataCompressBin] = bit ^ 1;
    sizeDataCompressBin = sizeDataCompressBin + 1;
  }
}


void ArithmeticEncoder::finish() {
  dataCompressBin[sizeDataCompressBin] = 1;
  sizeDataCompressBin = sizeDataCompressBin + 1;
}


void ArithmeticEncoder::underflow() {
  numUnderflow++;
}

/*
 * GETTER
 */


uint8_t* ArithmeticEncoder::getDataCompressBin()
{
  return dataCompressBin;
}

int ArithmeticEncoder::getSizeDataCompressBin()
{
  return sizeDataCompressBin;
}

uint8_t* ArithmeticEncoder::getDataCompressHex()
{
  return dataCompressHex;
}

int ArithmeticEncoder::getSizeDataCompressHex()
{
  return sizeDataCompressHex;
}
