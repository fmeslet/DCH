#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <Arduino.h>
#include "context_mapper.h"


class Compressor
{
  private:
    uint8_t* dataCompressBin; //[1000];
    int sizeDataCompressBin;
    
    uint8_t dataCompressHex[1000];
    int sizeDataCompressHex;
    int alphabetSize;

    ContextMapper* contextMapper;

    // Constants
    int timesteps;
    int sizePacketBin;
    int sizePacketHex;
    int sizeContext;

  public:
    // Constructor
    Compressor(int alphabetS, 
               ContextMapper* contextM,
               int timeS,
               int sizePacketB,
               int sizePacketH,
               int sizeC);

    // Methods basics
    void compress(uint8_t* dataHex,
                  int sizeArrayHex,
                  uint8_t* keyArray,
                  int sizeKeyArray);
    /*void decompress(uint8_t* dataHex,
                  int sizeArrayHex,
                  uint8_t* keyArray,
                  int sizeKeyArray);*/

    // Others
    void convertBinToHex(uint8_t* arrayBin, // /!\ or take bitset
                         int sizeArrayBin,
                         uint8_t* arrayHex);
    void convertHexToBin(uint8_t* arrayHex, // /!\ or return bitset
                          int sizeArrayHex,
                          uint8_t* arrayBin);
       

    // Getter and setter
    void setCumul(float* freqs, 
                  uint64_t* cumul);
                  
    uint8_t* getDataCompressBin();
    int getSizeDataCompressBin();
    uint8_t* getDataCompressHex();
    int getSizeDataCompressHex();
    
};


#endif
